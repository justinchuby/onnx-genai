//! Typed native capability ledger for the CPU execution provider.
//!
//! # Why this exists
//!
//! **The production CPU EP is native.** A default build links no MLAS, exports
//! no MLAS, and routes nothing to it;
//! `crates/onnx-runtime-ep-cpu-plugin/tests/default_artifacts_are_mlas_free.rs`
//! falsifies that claim on the shipped cdylib rather than restating it. Nor is
//! there any fallback to ORT's built-in `CPUExecutionProvider` — there is no
//! variant of [`Backend`] that could express it, and
//! `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs` proves the
//! runtime behaviour with `session.disable_cpu_ep_fallback=1`.
//!
//! MLAS is a **research reference**, reachable only in a `--features mlas`
//! test/benchmark build. Its role is to be measured against and absorbed:
//! replace each family with a native kernel that is at least as correct and
//! measurably faster (see `docs/performance/CPU_MLAS_MIGRATION.md`). That
//! absorption needs a written, machine-checkable answer to two questions, per
//! kernel family:
//!
//! 1. **What does the native EP run today, and where is a reference still
//!    ahead?** [`PLAN`] records `Native`, `Mlas`, or `NativeOverMlas` for every
//!    [`KernelFamily`], with the dtype / ISA / thread-model / shape-gate
//!    evidence behind that statement. `Mlas` here means "a research build can
//!    reach MLAS for this family, and absorbing it is outstanding work" — never
//!    "production ships MLAS here".
//! 2. **What actually ran?** [`record`] captures live [`Observation`]s so a
//!    test or a benchmark can assert the route it believes it measured.
//!    Recording is **off unless asked for** (`NXRT_CPU_DISPATCH_LEDGER=1`, or
//!    [`enable`]), so a production decode loop pays one relaxed atomic load per
//!    route decision — provided dispatch sites use [`record_with`], which does
//!    not build the evidence unless it will be kept — and the retained log is
//!    bounded by [`MAX_OBSERVATIONS`].
//!
//! # Reading the plan
//!
//! [`PLAN`] is the *declared* route. [`effective_backend`] is what this build
//! can actually reach: in a default build — every shipped build — each family
//! degrades to [`Backend::Native`], because MLAS is simply not linked. A family
//! whose plan is [`Backend::Native`] is **graduated**: no reference is ahead of
//! us there any more, and the entry records the measurement that earned it.

use core::fmt;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Environment variable that turns live [`Observation`] recording on.
pub const LEDGER_ENV: &str = "NXRT_CPU_DISPATCH_LEDGER";

// ─── Types ──────────────────────────────────────────────────────────────────

/// A family of kernels that share one backend decision.
///
/// Families are the unit of the MLAS migration: an absorption graduates a whole
/// family, not a single shape. Ordering here is the ordering used by
/// `docs/performance/CPU_MLAS_MIGRATION.md`'s replacement priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum KernelFamily {
    /// Dense f32 `MatMul` / batched `MatMul`.
    MatMulF32,
    /// `Gemm` (alpha/beta/transpose/bias) in f32.
    GemmF32,
    /// Block-quantized `MatMulNBits` (int4 / int8, prefill and decode).
    MatMulNBits,
    /// `QLinearMatMul` / `MatMulInteger` u8×u8 and u8×i8 integer GEMM.
    QLinearMatMul,
    /// Elementwise transcendental activations: SiLU, GELU, Tanh, Sigmoid, Erf.
    Activations,
    /// `Softmax` / `LogSoftmax` row reductions.
    Softmax,
    /// `LayerNormalization` / `RMSNormalization` / `SkipLayerNormalization`.
    Normalization,
    /// Scaled dot-product attention and its transposes (`SDPA`, `GQA`, `MHA`).
    AttentionTranspose,
    /// Quantize / dequantize / requantize element paths.
    Quantization,
    /// Block-quantized mixture-of-experts (`QMoE`).
    MoE,
    /// f32 `Conv` and the NCHWc blocked-layout convolution path.
    Convolution,
    /// `MaxPool` / `AveragePool`, including the NCHWc blocked variants.
    Pooling,
    /// Dense elementwise binary/unary ops (`Add`, `Mul`, `Clip`, `Relu`, …).
    Elementwise,
}

impl KernelFamily {
    /// Every family, in migration-priority order. Used by the contract tests to
    /// prove [`PLAN`] is total.
    pub const ALL: &'static [KernelFamily] = &[
        KernelFamily::MatMulF32,
        KernelFamily::GemmF32,
        KernelFamily::MatMulNBits,
        KernelFamily::QLinearMatMul,
        KernelFamily::Activations,
        KernelFamily::Softmax,
        KernelFamily::Normalization,
        KernelFamily::AttentionTranspose,
        KernelFamily::Quantization,
        KernelFamily::MoE,
        KernelFamily::Convolution,
        KernelFamily::Pooling,
        KernelFamily::Elementwise,
    ];

    /// Stable snake_case name, used in ledger dumps and test assertions.
    pub const fn name(self) -> &'static str {
        match self {
            KernelFamily::MatMulF32 => "matmul_f32",
            KernelFamily::GemmF32 => "gemm_f32",
            KernelFamily::MatMulNBits => "matmul_nbits",
            KernelFamily::QLinearMatMul => "qlinear_matmul",
            KernelFamily::Activations => "activations",
            KernelFamily::Softmax => "softmax",
            KernelFamily::Normalization => "normalization",
            KernelFamily::AttentionTranspose => "attention_transpose",
            KernelFamily::Quantization => "quantization",
            KernelFamily::MoE => "moe",
            KernelFamily::Convolution => "convolution",
            KernelFamily::Pooling => "pooling",
            KernelFamily::Elementwise => "elementwise",
        }
    }
}

impl fmt::Display for KernelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Which implementation ran, **inside this execution provider**.
///
/// There is deliberately no `OrtCpuEp` variant. Handing a node to ORT's own CPU
/// EP is not a backend choice this EP can make; a node we claim, we run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Backend {
    /// Our own kernel end to end. No MLAS symbol is on the hot path.
    Native,
    /// The vendored MLAS kernel does the arithmetic; we own partitioning,
    /// threading and the node contract around it.
    Mlas,
    /// Our kernel supplies the outer structure (tiling, threading, epilogue,
    /// fusion) and calls MLAS for an inner primitive. Absorbing the inner
    /// primitive graduates the family to [`Backend::Native`].
    NativeOverMlas,
}

impl Backend {
    /// Stable snake_case name.
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Native => "native",
            Backend::Mlas => "mlas",
            Backend::NativeOverMlas => "native_over_mlas",
        }
    }

    /// Whether this choice has MLAS anywhere on its hot path.
    pub const fn uses_mlas(self) -> bool {
        matches!(self, Backend::Mlas | Backend::NativeOverMlas)
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Where a family stands in the MLAS migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Graduation {
    /// A native kernel owns the hot path; MLAS is not called. The string is the
    /// measurement that earned the graduation.
    Graduated(&'static str),
    /// MLAS is the baseline. A native replacement must clear the graduation
    /// rule in `docs/performance/CPU_MLAS_MIGRATION.md` before it takes over.
    MlasBaseline,
    /// A native kernel already owns part of the family; the rest still routes
    /// through MLAS.
    Partial(&'static str),
    /// MLAS has no primitive for this family, so the question never arises.
    NoMlasPrimitive,
}

impl Graduation {
    /// Stable snake_case discriminant name, without the evidence payload.
    pub const fn name(self) -> &'static str {
        match self {
            Graduation::Graduated(_) => "graduated",
            Graduation::MlasBaseline => "mlas_baseline",
            Graduation::Partial(_) => "partial",
            Graduation::NoMlasPrimitive => "no_mlas_primitive",
        }
    }
}

/// One family's declared route and the evidence behind it.
///
/// Every field is prose on purpose: this is a ledger a reviewer reads, and a
/// machine-readable shape gate would be a second source of truth for something
/// the kernel already decides. What the tests enforce is that the ledger is
/// *total*, *unique*, and *consistent with the build* — not that it restates
/// the kernel's arithmetic.
/// How a family's MLAS route is *reached*, which decides where it exists.
///
/// The `mlas` feature is not the whole gate. Two different mechanisms call into
/// MLAS in this crate and they have different reachability:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteGate {
    /// The kernel calls `mlas_sys::` directly under `#[cfg(feature = "mlas")]`.
    /// MLAS ships NEON kernels as well as x86 ones, so these routes exist on
    /// every target the crate builds for.
    WhereverLinked,
    /// The route is selected through [`crate::backend::CpuBackend::auto_detect`],
    /// which returns `Mlas` only on x86-64, and only when the host is not
    /// Android (`Xnnpack`) or Apple (`Accelerate`). On aarch64 Linux, Windows
    /// ARM64 and macOS these families run a **native** GEMM no matter what the
    /// feature says, and the ledger has to report that rather than the plan.
    GemmBackend,
}

/// Whether [`crate::backend::CpuBackend::auto_detect`] can return `Mlas` here.
///
/// Kept as a `const fn` mirroring `auto_detect`'s own `cfg` arms so the ledger
/// and the dispatcher cannot disagree; `gemm_families_report_the_reachable_route`
/// asserts they do not.
pub const fn gemm_backend_is_mlas() -> bool {
    cfg!(all(
        feature = "mlas",
        target_arch = "x86_64",
        not(target_os = "android"),
        not(target_os = "macos"),
        not(target_os = "ios")
    ))
}

#[derive(Debug, Clone, Copy)]
pub struct PlanEntry {
    /// The family this entry governs.
    pub family: KernelFamily,
    /// The route taken when the `mlas` feature is compiled in (the default).
    pub planned: Backend,
    /// How the MLAS route is reached, and therefore on which targets it exists.
    pub gate: RouteGate,
    /// Element types this route covers.
    pub dtypes: &'static str,
    /// Instruction-set evidence: what the route dispatches on at runtime.
    pub isa: &'static str,
    /// Thread model: who partitions the work and which pool runs it.
    pub threads: &'static str,
    /// Shape evidence: the gate that picks this route over the alternative.
    pub shape_gate: &'static str,
    /// Migration status.
    pub graduation: Graduation,
}

impl PlanEntry {
    /// The route this *build*, on this *target*, can actually reach.
    ///
    /// Two ways a plan degrades to [`Backend::Native`]:
    ///
    /// * the `mlas` feature is off, so MLAS is not linked at all; or
    /// * the family is reached through [`RouteGate::GemmBackend`] and this
    ///   target is one where `auto_detect` never returns `Mlas` — aarch64,
    ///   Apple, or Android. A ledger that reported `Mlas` for `MatMulF32` on
    ///   Windows ARM64 would be stating the plan while the binary ran a native
    ///   GEMM, which is the drift this module exists to prevent.
    ///
    /// The native path is always present: it is the correctness baseline every
    /// MLAS route is differentially tested against.
    pub const fn effective(&self) -> Backend {
        if !cfg!(feature = "mlas") {
            return Backend::Native;
        }
        match self.gate {
            RouteGate::WhereverLinked => self.planned,
            RouteGate::GemmBackend if gemm_backend_is_mlas() => self.planned,
            RouteGate::GemmBackend => Backend::Native,
        }
    }
}

/// The declared route for every [`KernelFamily`].
///
/// Keep this ordered as [`KernelFamily::ALL`] and in step with
/// `docs/performance/CPU_MLAS_MIGRATION.md`; `plan_is_total_and_unique` and
/// `plan_order_matches_family_order` fail the build otherwise.
pub const PLAN: &[PlanEntry] = &[
    PlanEntry {
        family: KernelFamily::MatMulF32,
        planned: Backend::Mlas,
        gate: RouteGate::GemmBackend,
        dtypes: "f32 (f16/bf16 widen to f32 and take the native blocked path)",
        isa: "MLAS runtime dispatch: AVX-512 / AVX2+FMA / SSE2 on x86-64, NEON on aarch64; \
              native SimdX86 needs AVX2+FMA, native Generic needs nothing",
        threads: "MLAS partitions; tiles run on the mlas-sys work-stealing pool under the EP \
                  thread budget",
        shape_gate: "all shapes on x86-64; NXRT_CPU_GEMM_BACKEND=generic|simd|mlas overrides. \
                     The native SimdX86 route sends M=1 to a dedicated GEMV that streams B \
                     in place instead of packing panels that are reused zero times (#1091, \
                     ported in #1116, shipped on by default in #1183). It is a compile-time \
                     route, not an env toggle: sgemm_simd always passes use_m1_gemv=true, \
                     and only the in-process A/B harness passes false to measure the packed \
                     path it replaced",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::GemmF32,
        planned: Backend::NativeOverMlas,
        gate: RouteGate::GemmBackend,
        dtypes: "f32",
        isa: "same MLAS float dispatch as MatMulF32",
        threads: "MLAS partitions the GEMM; alpha/beta/bias epilogue is ours, single pass",
        shape_gate: "alpha==1 && beta in {0,1} routes the GEMM to MLAS; other scalings stay native",
        graduation: Graduation::Partial(
            "the epilogue is already ours; the inner SGEMM is the MatMulF32 dependency",
        ),
    },
    PlanEntry {
        family: KernelFamily::MatMulNBits,
        planned: Backend::NativeOverMlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "int4 / int8 weights, f32 activations, f16 scales",
        isa: "native: AVX2/AVX-512/VNNI and NEON dot; MLAS SQNBit: its own dispatch",
        threads: "ours — the persistent SPMD decode pool shards N; MLAS is called per shard",
        shape_gate: "M=1 decode is native (absorbed, #1104); M>=NXRT_SQNBIT_DECODE_MIN \
                     routes to MLAS SQNBit with a packed-B copy",
        graduation: Graduation::Partial(
            "int4 acc0 decode absorbed at 1.36x/1.56x vs the old borrowed path (#1104); \
             prefill still MLAS",
        ),
    },
    PlanEntry {
        family: KernelFamily::QLinearMatMul,
        planned: Backend::Mlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "u8xu8, u8xi8 integer GEMM with i32 accumulation",
        isa: "MLAS QGemm dispatch (AVX2 / AVX-VNNI / AVX-512 VNNI, NEON udot/sdot)",
        threads: "MLAS partitions; requantization runs inside MLAS the way ORT does (#1125)",
        shape_gate: "all shapes where MLAS reports the zero-point/signedness combination exact; \
                     otherwise the native i32 reference runs",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::Activations,
        planned: Backend::NativeOverMlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "f32 (f16/bf16 widen)",
        isa: "MLAS SIMD transcendentals; native NEON on aarch64, native scalar elsewhere",
        threads: "ours — run_chunked shards across the EP pool, MLAS is single-threaded per chunk",
        shape_gate: "len >= SIMD_MIN_LEN takes MLAS; shorter runs stay native; SiLU outside \
                     +/-18 and GELU at -inf are repaired natively because MLAS clamps",
        graduation: Graduation::Partial(
            "chunking, special-value repair and threading are ours; the inner kernel is MLAS",
        ),
    },
    PlanEntry {
        family: KernelFamily::Softmax,
        planned: Backend::Mlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "f32",
        isa: "MLAS SIMD exp/reduce dispatch",
        threads: "ours — rows are sharded across the EP pool, MLAS runs a shard serially",
        shape_gate: "contiguous last-axis rows; strided/other-axis cases stay native",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::Normalization,
        planned: Backend::Native,
        gate: RouteGate::WhereverLinked,
        dtypes: "f32 / f16 / bf16",
        isa: "native SIMD (AVX2/AVX-512, NEON) with f32 accumulation",
        threads: "ours — rows sharded across the EP pool",
        shape_gate: "all shapes",
        graduation: Graduation::NoMlasPrimitive,
    },
    PlanEntry {
        family: KernelFamily::AttentionTranspose,
        planned: Backend::NativeOverMlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "f32 / f16 / bf16",
        isa: "native attention loops; MLAS SGEMM for the QK^T and PV products",
        threads: "ours — heads and sequence tiles are sharded by the EP pool",
        shape_gate: "batched f32 QK^T/PV go through MlasGemmBatch; masking, softmax fusion and \
                     the KV layout are native",
        graduation: Graduation::Partial("everything except the two inner GEMMs is native"),
    },
    PlanEntry {
        family: KernelFamily::Quantization,
        planned: Backend::NativeOverMlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "f32 <-> u8/i8/int4",
        isa: "MLAS quantize/dequantize SIMD; native block dequant for int4",
        threads: "ours",
        shape_gate: "per-tensor and per-column u8/i8 use MLAS; block/int4 formats are native",
        graduation: Graduation::Partial("block-quantized formats are already native"),
    },
    PlanEntry {
        family: KernelFamily::MoE,
        planned: Backend::NativeOverMlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "int4 / int8 expert weights, f32 activations",
        isa: "inherits MatMulNBits dispatch",
        threads: "ours — experts are sharded across the EP pool",
        shape_gate: "routing, gating and expert gather are native; the per-expert GEMM follows \
                     the MatMulNBits gate",
        graduation: Graduation::Partial("blocked on the MatMulNBits prefill absorption"),
    },
    PlanEntry {
        family: KernelFamily::Convolution,
        planned: Backend::Mlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "f32",
        isa: "MLAS NCHWc blocked convolution and its im2col/direct kernels",
        threads: "MLAS partitions; runs on the mlas-sys pool",
        shape_gate: "with `mlas`, kernels/conv.rs plus the Nchwc reorder ops are registered; \
                     without it, conv_ref.rs (native reference) is registered instead",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::Pooling,
        planned: Backend::Mlas,
        gate: RouteGate::WhereverLinked,
        dtypes: "f32",
        isa: "MLAS pooling kernels, including the NCHWc blocked variants",
        threads: "MLAS partitions",
        shape_gate: "MaxPool/AveragePool in f32; other dtypes stay native",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::Elementwise,
        planned: Backend::Native,
        gate: RouteGate::WhereverLinked,
        dtypes: "all supported dtypes",
        isa: "native dense SIMD dispatch with a scalar fallback",
        threads: "ours — dense ranges sharded across the EP pool",
        shape_gate: "dense inputs take the SIMD path; strided inputs take the general path. \
                     MlasEltwiseAdd measured no better than ours, so f32 Add stays native",
        graduation: Graduation::Graduated(
            "native dense SIMD matched or beat MlasEltwiseAdd on f32 Add; MLAS is not called",
        ),
    },
];

/// The declared plan for `family`.
///
/// Infallible: [`PLAN`] is proved total over [`KernelFamily::ALL`] by
/// `plan_is_total_and_unique`.
pub fn plan_for(family: KernelFamily) -> &'static PlanEntry {
    PLAN.iter()
        .find(|entry| entry.family == family)
        .expect("PLAN is total over KernelFamily::ALL (see plan_is_total_and_unique)")
}

/// The backend this build actually reaches for `family`.
pub fn effective_backend(family: KernelFamily) -> Backend {
    plan_for(family).effective()
}

/// Whether MLAS is linked into this build of the CPU EP.
pub const fn mlas_linked() -> bool {
    cfg!(feature = "mlas")
}

// ─── Live observations ──────────────────────────────────────────────────────

/// Instruction set a route dispatched to, as observed at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Isa {
    /// x86-64 with AVX-512F.
    Avx512,
    /// x86-64 with AVX2 and FMA.
    Avx2Fma,
    /// x86-64 without AVX2/FMA (SSE2 baseline).
    SseBaseline,
    /// aarch64 Advanced SIMD.
    Neon,
    /// No SIMD claim: a scalar route, or an architecture we do not probe.
    Scalar,
}

impl Isa {
    /// The best ISA this host offers for f32 kernels.
    pub fn host() -> Isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                Isa::Avx512
            } else if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                Isa::Avx2Fma
            } else {
                Isa::SseBaseline
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Isa::Neon
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Isa::Scalar
        }
    }

    /// Stable snake_case name.
    pub const fn name(self) -> &'static str {
        match self {
            Isa::Avx512 => "avx512",
            Isa::Avx2Fma => "avx2_fma",
            Isa::SseBaseline => "sse_baseline",
            Isa::Neon => "neon",
            Isa::Scalar => "scalar",
        }
    }
}

/// One recorded routing decision, with the evidence that explains it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    /// Family that dispatched.
    pub family: KernelFamily,
    /// Implementation that ran.
    pub backend: Backend,
    /// Element type of the dominant operand, as an ONNX dtype name.
    pub dtype: &'static str,
    /// Instruction set the route dispatched to.
    pub isa: Isa,
    /// Degree of parallelism the route was given.
    pub threads: usize,
    /// `(M, N, K)` for GEMM-shaped families, or `(elements, 1, 1)` for
    /// elementwise ones. Zero means "not applicable".
    pub shape: (usize, usize, usize),
}

impl Observation {
    /// A GEMM-shaped observation.
    pub fn gemm(
        family: KernelFamily,
        backend: Backend,
        dtype: &'static str,
        m: usize,
        n: usize,
        k: usize,
    ) -> Observation {
        Observation {
            family,
            backend,
            dtype,
            isa: Isa::host(),
            threads: thread_degree(),
            shape: (m, n, k),
        }
    }

    /// An elementwise observation over `elements` values.
    pub fn elementwise(
        family: KernelFamily,
        backend: Backend,
        dtype: &'static str,
        elements: usize,
    ) -> Observation {
        Observation {
            family,
            backend,
            dtype,
            isa: Isa::host(),
            threads: thread_degree(),
            shape: (elements, 1, 1),
        }
    }
}

/// Parallelism the EP currently offers, for observation evidence.
fn thread_degree() -> usize {
    #[cfg(feature = "mlas")]
    {
        mlas_sys::mlas_threading_degree()
    }
    #[cfg(not(feature = "mlas"))]
    {
        rayon::current_num_threads()
    }
}

/// Recording state: [`UNINIT`] until the environment has been consulted, then
/// [`OFF`] or [`ON`].
///
/// Tri-state rather than a `bool` behind a `OnceLock` so that the steady-state
/// read on a dispatch path is exactly one relaxed load and a branch. A `OnceLock`
/// guard would add an acquire load and an opaque call to every route decision
/// forever, to answer a question that is settled after the first one.
static RECORDING: AtomicU8 = AtomicU8::new(UNINIT);

const UNINIT: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

/// Hard upper bound on retained [`Observation`]s.
///
/// The log is a diagnostic, not a trace: what a caller asks it is "did family
/// X reach backend Y", which the first `N` observations answer as well as an
/// unbounded `Vec` does. Without a cap, `NXRT_CPU_DISPATCH_LEDGER=1` on a
/// long-running server grows the log by one `Observation` per route decision
/// for the life of the process, with `reset()` the only thing that shrinks it
/// — a retained allocation whose multiplier (the dispatch count) is unstated,
/// which is the exact shape of four prior defects in this tree. `Observation`
/// is `Copy`-sized, so 64 Ki of them is a bounded few MiB.
pub const MAX_OBSERVATIONS: usize = 65_536;

fn log() -> &'static Mutex<Vec<Observation>> {
    static LOG: OnceLock<Mutex<Vec<Observation>>> = OnceLock::new();
    LOG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Observations dropped because the log was already at [`MAX_OBSERVATIONS`].
///
/// Non-zero means [`snapshot`] is a prefix, not the whole run. Reported rather
/// than silently absorbed, so a test cannot conclude "family X never reached
/// backend Y" from a truncated log.
static DROPPED: AtomicUsize = AtomicUsize::new(0);

/// Honour `NXRT_CPU_DISPATCH_LEDGER=1` on the first read.
///
/// Off the hot path: reached once per process, or never if [`enable`] or
/// [`disable`] settles the state first. Two threads racing here read the same
/// environment and compute the same answer, and the compare-exchange keeps
/// either of them from overwriting an explicit [`enable`] that landed in
/// between.
#[cold]
#[inline(never)]
fn init_from_env() -> bool {
    let from_env = std::env::var(LEDGER_ENV).as_deref() == Ok("1");
    let settled = if from_env { ON } else { OFF };
    match RECORDING.compare_exchange(UNINIT, settled, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => from_env,
        Err(already) => already == ON,
    }
}

/// Whether live recording is on.
///
/// One relaxed load and a branch once the state has settled, which is what a
/// dispatch site pays for having the ledger available but off.
#[inline]
pub fn is_recording() -> bool {
    match RECORDING.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => init_from_env(),
    }
}

/// Turn live recording on. Intended for tests and benchmarks.
///
/// Supersedes the environment, and settles the state if nothing has read it
/// yet, so a later first read cannot revert this.
pub fn enable() {
    RECORDING.store(ON, Ordering::Relaxed);
}

/// Turn live recording off.
pub fn disable() {
    RECORDING.store(OFF, Ordering::Relaxed);
}

/// Drop every recorded observation.
pub fn reset() {
    if let Ok(mut entries) = log().lock() {
        entries.clear();
        // Under the same lock `record` takes, so a concurrent drop cannot be
        // credited to the run that was just cleared.
        DROPPED.store(0, Ordering::Relaxed);
    }
}

/// How many observations were discarded because the log hit
/// [`MAX_OBSERVATIONS`] since the last [`reset`].
pub fn dropped() -> usize {
    DROPPED.load(Ordering::Relaxed)
}

/// Record a routing decision.
///
/// Takes an already-built [`Observation`], so the caller pays for gathering the
/// evidence whether or not it is kept. On a kernel dispatch path use
/// [`record_with`] instead; this form is for tests and for callers that already
/// hold an observation. Call it where the route is *chosen*, not in an inner
/// loop — and note that "call it in the right place" is guidance, whereas
/// [`MAX_OBSERVATIONS`] is the bound.
#[inline]
pub fn record(observation: Observation) {
    if !is_recording() {
        return;
    }
    if let Ok(mut entries) = log().lock() {
        if entries.len() >= MAX_OBSERVATIONS {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        entries.push(observation);
    }
}

/// Record an observation that is only *built* if recording is on.
///
/// Prefer this at kernel dispatch sites. `record(Observation::gemm(..))`
/// evaluates its argument first, and building an [`Observation`] probes the
/// host ISA and asks MLAS for its thread degree — an FFI call — so the
/// "one relaxed atomic load when off" cost this module advertises was not what
/// a production decode loop actually paid. With the closure, an off ledger
/// costs the load and nothing else.
#[inline]
pub fn record_with(observation: impl FnOnce() -> Observation) {
    if !is_recording() {
        return;
    }
    record(observation());
}

/// Every observation recorded since the last [`reset`].
pub fn snapshot() -> Vec<Observation> {
    log()
        .lock()
        .map(|entries| entries.clone())
        .unwrap_or_default()
}

/// Whether any observation for `family` recorded `backend`.
pub fn observed(family: KernelFamily, backend: Backend) -> bool {
    snapshot()
        .iter()
        .any(|o| o.family == family && o.backend == backend)
}

/// A human-readable dump of [`PLAN`] against this build, for `--nocapture`
/// test output and benchmark headers.
pub fn render_plan() -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "CPU EP dispatch ledger (mlas linked: {}, host ISA: {})\n",
        mlas_linked(),
        Isa::host().name()
    ));
    for entry in PLAN {
        out.push_str(&format!(
            "  {:<20} planned={:<17} effective={:<17} status={}\n      dtypes: {}\n      isa: {}\n      threads: {}\n      shape gate: {}\n",
            entry.family.name(),
            entry.planned.name(),
            entry.effective().name(),
            entry.graduation.name(),
            entry.dtypes,
            entry.isa,
            entry.threads,
            entry.shape_gate,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises the tests that toggle the process-global recorder. Two of
    /// them running concurrently would race each other rather than the code,
    /// and the resulting flake would look like a ledger defect.
    static RECORDER_TESTS: Mutex<()> = Mutex::new(());

    /// Take [`RECORDER_TESTS`], ignoring a poisoned lock: a panic in one
    /// recorder test must not convert every other one into a second failure
    /// that hides it.
    fn recorder_guard() -> std::sync::MutexGuard<'static, ()> {
        RECORDER_TESTS.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn plan_is_total_and_unique() {
        assert_eq!(
            PLAN.len(),
            KernelFamily::ALL.len(),
            "PLAN must have exactly one entry per KernelFamily"
        );
        for family in KernelFamily::ALL {
            let matches = PLAN.iter().filter(|e| e.family == *family).count();
            assert_eq!(matches, 1, "{family} must appear exactly once in PLAN");
        }
    }

    #[test]
    fn plan_order_matches_family_order() {
        let planned: Vec<KernelFamily> = PLAN.iter().map(|e| e.family).collect();
        assert_eq!(
            planned,
            KernelFamily::ALL.to_vec(),
            "PLAN must stay in KernelFamily::ALL (migration-priority) order"
        );
    }

    #[test]
    fn family_names_are_unique() {
        let mut names: Vec<&str> = KernelFamily::ALL.iter().map(|f| f.name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "family names must be unique");
    }

    /// Without MLAS linked, no family may claim an MLAS route: the symbols are
    /// not there. This is what makes `--no-default-features` a real opt-out
    /// rather than a label.
    ///
    /// With MLAS linked the plan holds only for [`RouteGate::WhereverLinked`]
    /// families. The GEMM ones degrade on aarch64, Apple and Android even in a
    /// full MLAS build, because `auto_detect` never offers them `Mlas` there —
    /// `gemm_families_report_the_reachable_route` owns that half.
    #[test]
    fn effective_route_degrades_without_mlas() {
        for entry in PLAN {
            if !mlas_linked() {
                assert_eq!(
                    entry.effective(),
                    Backend::Native,
                    "{} must degrade to native without the mlas feature",
                    entry.family
                );
            } else if entry.gate == RouteGate::WhereverLinked {
                assert_eq!(
                    entry.effective(),
                    entry.planned,
                    "{} calls mlas-sys directly, so a linked build always reaches it",
                    entry.family
                );
            }
        }
    }

    /// The ledger's GEMM verdict must be the dispatcher's, on this target.
    ///
    /// `effective()` used to gate on the `mlas` feature alone, which made it
    /// claim `Mlas` for `MatMulF32` on aarch64, Windows ARM64 and macOS —
    /// targets where `auto_detect` returns `Generic` or `Accelerate` and MLAS
    /// is never asked for a GEMM. A ledger that reports the plan instead of the
    /// binary is worse than no ledger, so this compares the two directly rather
    /// than restating `gemm_backend_is_mlas()`.
    #[test]
    fn gemm_families_report_the_reachable_route() {
        let dispatcher_uses_mlas = {
            #[cfg(feature = "mlas")]
            {
                crate::backend::CpuBackend::auto_detect() == crate::backend::CpuBackend::Mlas
            }
            #[cfg(not(feature = "mlas"))]
            {
                false
            }
        };
        assert_eq!(
            gemm_backend_is_mlas(),
            dispatcher_uses_mlas,
            "the ledger's cfg mirror of auto_detect has drifted from auto_detect itself"
        );

        for entry in PLAN.iter().filter(|e| e.gate == RouteGate::GemmBackend) {
            let reaches_mlas = entry.effective() != Backend::Native;
            assert_eq!(
                reaches_mlas,
                dispatcher_uses_mlas,
                "{} reports {:?} but auto_detect gives {:?} on this target",
                entry.family.name(),
                entry.effective(),
                crate::backend::CpuBackend::auto_detect()
            );
        }
    }

    /// Only the families that actually go through `CpuBackend` may claim that
    /// gate. Everything else calls `mlas_sys::` directly and is reachable on
    /// every target, so mislabelling one would silently suppress its MLAS route
    /// from the ledger on ARM.
    #[test]
    fn only_the_cpu_backend_families_use_the_gemm_gate() {
        let gated: Vec<&str> = PLAN
            .iter()
            .filter(|e| e.gate == RouteGate::GemmBackend)
            .map(|e| e.family.name())
            .collect();
        assert_eq!(gated, vec!["matmul_f32", "gemm_f32"]);
    }

    /// A graduated family has absorbed MLAS; it must not still plan an MLAS
    /// route. This is the invariant that keeps the ledger honest as families
    /// migrate.
    #[test]
    fn graduated_families_do_not_plan_mlas() {
        for entry in PLAN {
            match entry.graduation {
                Graduation::Graduated(evidence) => {
                    assert_eq!(
                        entry.planned,
                        Backend::Native,
                        "{} is marked graduated but still plans {}",
                        entry.family,
                        entry.planned
                    );
                    assert!(
                        !evidence.is_empty(),
                        "{} must cite the measurement that graduated it",
                        entry.family
                    );
                }
                Graduation::NoMlasPrimitive => assert_eq!(
                    entry.planned,
                    Backend::Native,
                    "{} claims MLAS has no primitive but plans {}",
                    entry.family,
                    entry.planned
                ),
                Graduation::MlasBaseline => assert!(
                    entry.planned.uses_mlas(),
                    "{} is the MLAS baseline but plans {}",
                    entry.family,
                    entry.planned
                ),
                Graduation::Partial(evidence) => {
                    assert_eq!(
                        entry.planned,
                        Backend::NativeOverMlas,
                        "{} is partially absorbed, so its plan must be native_over_mlas",
                        entry.family
                    );
                    assert!(
                        !evidence.is_empty(),
                        "{} must say what is already native",
                        entry.family
                    );
                }
            }
        }
    }

    #[test]
    fn every_entry_carries_evidence() {
        for entry in PLAN {
            for (label, text) in [
                ("dtypes", entry.dtypes),
                ("isa", entry.isa),
                ("threads", entry.threads),
                ("shape_gate", entry.shape_gate),
            ] {
                assert!(
                    !text.trim().is_empty(),
                    "{}: {label} evidence is required",
                    entry.family
                );
            }
        }
    }

    /// Every environment variable the ledger's prose names must still be read
    /// somewhere in this crate's sources.
    ///
    /// This exists because it caught a real defect in this file. The
    /// `MatMulF32` entry advertised `ONNX_GENAI_CPU_MM_SIMD_M1_GEMV
    /// (default off)` long after #1183 had shipped that route on by default and
    /// deleted the probe, so the ledger — whose entire purpose is to describe
    /// where dispatch actually goes — was describing a knob that did not exist.
    /// `docs/performance/CPU_MATMUL_ASSIGNMENT.md` had recorded the correct
    /// fact the whole time; nothing compared the two.
    ///
    /// Prose cannot be type-checked, so this is a cheap falsifier rather than a
    /// complete one. Two limits are deliberate and worth stating, because an
    /// overclaimed guarantee is the thing this test exists to punish:
    ///
    /// * It checks that the named knob is still **read**, not that the
    ///   surrounding description of its behaviour is correct. Requiring the
    ///   name to appear in an `env::var` call or an `_ENV` constant — rather
    ///   than merely somewhere in the sources — is what keeps a name that
    ///   survives only inside a test's `EnvVarGuard` from passing as live.
    /// * It does not catch the inverse defect: a variable that genuinely gates
    ///   dispatch but that no `PlanEntry` mentions. Nothing here enumerates the
    ///   gates, so that direction needs a reader, not a test.
    #[test]
    fn ledger_prose_only_names_environment_variables_that_still_exist() {
        let source_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut sources = String::new();
        collect_rust_sources(&source_root, &mut sources);
        assert!(
            sources.len() > 10_000,
            "expected to have read the crate sources, got {} bytes -- if the \
             layout moved, fix this test rather than deleting it",
            sources.len()
        );

        for entry in PLAN {
            for text in [entry.dtypes, entry.isa, entry.threads, entry.shape_gate] {
                for name in environment_variable_names(text) {
                    let quoted = format!("\"{name}\"");
                    let read_sites: Vec<&str> = sources
                        .lines()
                        .filter(|line| line.contains(&quoted) && is_env_read_site(line))
                        .collect();
                    assert!(
                        !read_sites.is_empty(),
                        "{}: ledger prose names environment variable `{name}`, but no \
                         source file in this crate passes {quoted} to `env::var`/`var_os` \
                         or binds it to an `_ENV` constant. Occurrences in test guards do \
                         not count -- a name that only a test sets is not a live knob. \
                         Either the variable was retired (update the prose) or it moved \
                         crates (widen this test deliberately).",
                        entry.family
                    );
                }
            }
        }
    }

    /// Does this line actually wire the literal it contains up to the process
    /// environment? `env::set_var` is deliberately excluded: a test or bench
    /// that *sets* a variable is not evidence that anything reads it.
    fn is_env_read_site(line: &str) -> bool {
        line.contains("env::var(") || line.contains("env::var_os(") || line.contains(": &str =")
    }

    /// Concatenate every `.rs` file under `dir`, recursively.
    fn collect_rust_sources(dir: &std::path::Path, out: &mut String) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rust_sources(&path, out);
                continue;
            }
            if path.extension().is_some_and(|ext| ext == "rs") {
                out.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
            }
        }
    }

    /// Pull `SHOUTY_SNAKE_CASE` tokens that look like environment variables out
    /// of prose.
    ///
    /// The prefix list is the checked surface, not the crate's full set of
    /// knobs: the EP also reads `ONNX_RUNTIME_EP_CPU_*`, `EP_INTRA_OP` and
    /// `GEMM_AB_*`, none of which any `PlanEntry` currently names. Add a prefix
    /// here when a ledger entry starts naming one, rather than widening to all
    /// shouty words — ISA names like `AVX2` and `NEON` are not knobs.
    fn environment_variable_names(text: &str) -> Vec<String> {
        const CHECKED_PREFIXES: [&str; 3] = ["NXRT_", "ONNX_GENAI_", "ONNX_RUNTIME_EP_"];
        text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|token| {
                CHECKED_PREFIXES.iter().any(|p| token.starts_with(p)) && token.len() > 5
            })
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn recording_is_off_until_asked_for_and_bounded_once_it_is_on() {
        let _serialised = recorder_guard();
        // One test, because the recorder is process-global: two tests toggling
        // it in parallel would race each other rather than the code.
        //
        // It is global in a *third* sense that matters here — every other test
        // in this binary runs concurrently and its kernel dispatches record
        // too, once recording is on. So this asserts what is true regardless of
        // that traffic ("our observation is present", "the log is capped")
        // rather than exact counts, which would be a flake rather than a
        // check.
        disable();
        reset();
        record(Observation::gemm(
            KernelFamily::MatMulF32,
            Backend::Mlas,
            "f32",
            1,
            1,
            1,
        ));
        assert!(snapshot().is_empty(), "record() must no-op when disabled");

        enable();
        record(Observation::gemm(
            KernelFamily::MatMulF32,
            Backend::Native,
            "f32",
            4,
            8,
            16,
        ));
        let seen = snapshot();
        let ours = seen
            .iter()
            .find(|o| o.family == KernelFamily::MatMulF32 && o.shape == (4, 8, 16))
            .expect("the observation we just recorded must round-trip");
        assert_eq!(ours.backend, Backend::Native);
        assert_eq!(ours.isa, Isa::host());
        assert!(observed(KernelFamily::MatMulF32, Backend::Native));

        // The bound, falsified rather than restated: without the cap this loop
        // grows the log without limit, which is the defect it exists for.
        // Unbounded would give `len() > MAX_OBSERVATIONS` and `dropped() == 0`;
        // both assertions below are false in that world and true under any
        // amount of concurrent traffic.
        reset();
        let overshoot = 64;
        for i in 0..MAX_OBSERVATIONS + overshoot {
            record(Observation::gemm(
                KernelFamily::Softmax,
                Backend::Native,
                "f32",
                1,
                1,
                i,
            ));
        }
        assert_eq!(
            snapshot().len(),
            MAX_OBSERVATIONS,
            "the log must stop growing at MAX_OBSERVATIONS"
        );
        assert!(
            dropped() >= overshoot,
            "every discarded observation must be counted, so a truncated log \
             cannot be mistaken for a complete one; dropped={} overshoot={overshoot}",
            dropped()
        );
        // A truncated prefix still answers the question the ledger is asked.
        assert!(observed(KernelFamily::Softmax, Backend::Native));

        reset();
        assert_eq!(dropped(), 0, "reset clears the drop counter with the log");
        disable();
        reset();
    }

    #[test]
    fn render_plan_covers_every_family() {
        let rendered = render_plan();
        for family in KernelFamily::ALL {
            assert!(
                rendered.contains(family.name()),
                "render_plan omitted {family}"
            );
        }
    }

    /// The claim every dispatch site in this crate rests on: with the ledger
    /// off, [`record_with`] does not run its closure, so a shipped decode loop
    /// pays one relaxed load and nothing else.
    ///
    /// This is not a restatement of the code. Building an [`Observation`]
    /// probes the host ISA and asks for the thread degree — an FFI call in an
    /// MLAS build — so if the closure ran anyway, the "no measurable overhead
    /// when disabled" claim in the PR and in `CPU_MLAS_MIGRATION.md` would be
    /// false while every other test still passed. Counting closure entries is
    /// the only way to see the difference from inside the process.
    #[test]
    fn record_with_does_not_build_an_observation_while_disabled() {
        let _serialised = recorder_guard();
        disable();
        reset();

        let built = std::cell::Cell::new(0usize);
        // A fresh closure per call, so the count is the only thing carried
        // across iterations and nothing is moved into the loop.
        let build = |counter: &std::cell::Cell<usize>| {
            counter.set(counter.get() + 1);
            Observation::gemm(KernelFamily::GemmF32, Backend::Native, "f32", 2, 3, 5)
        };
        for _ in 0..10_000 {
            record_with(|| build(&built));
        }
        assert_eq!(
            built.get(),
            0,
            "a disabled ledger must not build the observation it is about to \
             throw away"
        );

        enable();
        record_with(|| build(&built));
        assert_eq!(
            built.get(),
            1,
            "an enabled ledger must build it exactly once"
        );
        assert!(
            snapshot()
                .iter()
                .any(|o| o.family == KernelFamily::GemmF32 && o.shape == (2, 3, 5)),
            "the observation built under `record_with` must reach the log"
        );

        disable();
        reset();
    }

    /// `record_with` is only cheaper than `record` if the *route decision* it
    /// guards is the same one. Both forms must land the same observation.
    #[test]
    fn record_and_record_with_agree_on_what_they_store() {
        let _serialised = recorder_guard();
        enable();
        reset();

        let direct = Observation::elementwise(KernelFamily::Softmax, Backend::Native, "f32", 4_096);
        record(direct);
        record_with(|| {
            Observation::elementwise(KernelFamily::Softmax, Backend::Native, "f32", 4_096)
        });
        let matching = snapshot().into_iter().filter(|o| *o == direct).count();
        assert!(
            matching >= 2,
            "record and record_with must store identical observations; saw \
             {matching} of them"
        );

        disable();
        reset();
    }

    /// [`plan_for`] and [`effective_backend`] are the ledger's public reading
    /// surface; a lookup that returned the wrong family, or a wrapper that
    /// disagreed with the entry it wraps, would misreport every route.
    #[test]
    fn public_lookups_agree_with_the_plan_entry() {
        for family in KernelFamily::ALL {
            let entry = plan_for(*family);
            assert_eq!(entry.family, *family, "plan_for returned another family");
            assert_eq!(
                effective_backend(*family),
                entry.effective(),
                "{family}: effective_backend disagrees with its own plan entry"
            );
            if !mlas_linked() {
                assert!(
                    !effective_backend(*family).uses_mlas(),
                    "{family}: a build with no MLAS linked cannot reach it"
                );
            }
        }
    }

    /// The names the ledger prints are its API for tests, dumps and the
    /// migration doc: they must be distinct, and `uses_mlas` must agree with
    /// what each variant means.
    #[test]
    fn backend_and_graduation_names_are_distinct_and_consistent() {
        let backends = [Backend::Native, Backend::Mlas, Backend::NativeOverMlas];
        let mut names: Vec<&str> = backends.iter().map(|b| b.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), backends.len(), "backend names must be unique");

        assert!(!Backend::Native.uses_mlas());
        assert!(Backend::Mlas.uses_mlas());
        assert!(
            Backend::NativeOverMlas.uses_mlas(),
            "a native outer loop over an MLAS inner primitive still has MLAS on \
             its hot path, which is the whole reason the variant exists"
        );

        assert_eq!(Graduation::Graduated("evidence").name(), "graduated");
        assert_eq!(Graduation::MlasBaseline.name(), "mlas_baseline");
        assert_eq!(Graduation::Partial("evidence").name(), "partial");
        assert_eq!(Graduation::NoMlasPrimitive.name(), "no_mlas_primitive");
    }

    /// An observation's evidence must describe the host it was taken on, and
    /// the two constructors must place their shapes where the field docs say.
    #[test]
    fn observations_carry_host_evidence_and_the_documented_shape() {
        let gemm = Observation::gemm(KernelFamily::MatMulF32, Backend::Native, "f32", 7, 11, 13);
        assert_eq!(gemm.shape, (7, 11, 13), "gemm shape is (M, N, K)");
        let ew = Observation::elementwise(KernelFamily::Activations, Backend::Native, "f32", 512);
        assert_eq!(
            ew.shape,
            (512, 1, 1),
            "elementwise shape is (elements, 1, 1)"
        );

        for o in [gemm, ew] {
            assert_eq!(o.isa, Isa::host());
            assert!(
                o.threads >= 1,
                "an observation must record a real thread degree, got {}",
                o.threads
            );
            assert!(!o.dtype.is_empty());
        }

        let mut isa_names: Vec<&str> = [
            Isa::Avx512,
            Isa::Avx2Fma,
            Isa::SseBaseline,
            Isa::Neon,
            Isa::Scalar,
        ]
        .iter()
        .map(|i| i.name())
        .collect();
        isa_names.sort_unstable();
        isa_names.dedup();
        assert_eq!(isa_names.len(), 5, "ISA names must be unique");

        // The host probe must agree with the architecture this build targets,
        // or the evidence names an ISA the code cannot have used.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(matches!(
            Isa::host(),
            Isa::Avx512 | Isa::Avx2Fma | Isa::SseBaseline
        ));
        #[cfg(target_arch = "aarch64")]
        assert_eq!(Isa::host(), Isa::Neon);
    }

    /// Every production dispatch site reaches the ledger through the *lazy*
    /// [`record_with`], never the eager [`record`].
    ///
    /// [`record_with_does_not_build_an_observation_while_disabled`] proves the
    /// primitive is lazy. It cannot prove the callers use it. Swapping any one
    /// site to `record(Observation::gemm(..))` would put [`Isa::host`] and
    /// [`thread_degree`] — an FFI call in an MLAS build — back on a hot path
    /// that is supposed to cost one relaxed load, and every other test in this
    /// crate would still pass. So the sites are asserted, by name, here.
    ///
    /// Reading the sources rather than the behaviour is deliberate: the
    /// difference between eager and lazy is invisible from outside the process
    /// once the observation is discarded, and there is no runtime hook that
    /// distinguishes them at a call site.
    #[test]
    fn production_dispatch_sites_record_lazily() {
        // (file, source, how many sites are expected to record)
        let sites: &[(&str, &str, usize)] = &[
            ("kernels/matmul.rs", include_str!("kernels/matmul.rs"), 1),
            (
                "kernels/simd_activations.rs",
                include_str!("kernels/simd_activations.rs"),
                2,
            ),
            ("kernels/softmax.rs", include_str!("kernels/softmax.rs"), 1),
        ];

        for (name, source, expected) in sites {
            let lazy = source.matches("record_with(").count();
            assert_eq!(
                lazy, *expected,
                "{name} must reach the ledger through `record_with` at exactly \
                 {expected} site(s); found {lazy}. If a dispatch site was added \
                 or removed on purpose, update the count here and say why."
            );

            // `record(` but not `record_with(` and not `..._record(`: the eager
            // form. Scanning for the bare identifier avoids matching
            // `record_with`, whose next byte is `_`.
            let eager: Vec<usize> = source
                .match_indices("record(")
                .filter(|(i, _)| {
                    let before = source[..*i].chars().next_back();
                    !matches!(before, Some(c) if c.is_alphanumeric() || c == '_')
                })
                .map(|(i, _)| source[..i].lines().count() + 1)
                .collect();
            assert!(
                eager.is_empty(),
                "{name} calls the eager `record` at line(s) {eager:?}. Production \
                 dispatch must use `record_with(|| ..)`, or building the \
                 observation costs an ISA probe and a thread-degree call on \
                 every dispatch even when the ledger is off."
            );
        }
    }
}
