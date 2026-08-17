//! Typed internal-backend dispatch ledger for the CPU execution provider.
//!
//! # Why this exists
//!
//! MLAS is compiled into this EP by default. That is a deliberate *internal*
//! choice: we keep full node ownership and call MLAS as a library we link, in
//! the same way we call [`rayon`] or [`libm`]. **Nothing in this crate ever
//! delegates a node to ORT's built-in `CPUExecutionProvider`** — there is no
//! variant of [`Backend`] that could express it, and
//! `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs` proves the
//! runtime behaviour with `session.disable_cpu_ep_fallback=1`.
//!
//! The long-term direction is to absorb MLAS: replace each family with a native
//! kernel that is at least as correct and measurably faster (see
//! `docs/performance/CPU_MLAS_MIGRATION.md`). That migration needs a written,
//! machine-checkable answer to two questions, per kernel family:
//!
//! 1. **What runs today?** [`PLAN`] records `Native`, `Mlas`, or
//!    `NativeOverMlas` for every [`KernelFamily`], with the dtype / ISA /
//!    thread-model / shape-gate evidence behind that choice.
//! 2. **What actually ran?** [`record`] captures live [`Observation`]s so a test
//!    or a benchmark can assert the route it believes it measured. Recording is
//!    **off unless asked for** (`NXRT_CPU_DISPATCH_LEDGER=1`, or [`enable`]), so
//!    a production decode loop pays one relaxed atomic load per route decision.
//!
//! # Reading the plan
//!
//! [`PLAN`] is the *declared* route. [`effective_backend`] is what this build
//! can actually reach: without the `mlas` Cargo feature every MLAS-bearing
//! family degrades to [`Backend::Native`], because MLAS is simply not linked.
//! A family whose plan is [`Backend::Native`] is **graduated** — MLAS is no
//! longer on its hot path and the entry records the measurement that earned it.

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
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
#[derive(Debug, Clone, Copy)]
pub struct PlanEntry {
    /// The family this entry governs.
    pub family: KernelFamily,
    /// The route taken when the `mlas` feature is compiled in (the default).
    pub planned: Backend,
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
    /// The route this *build* can actually reach.
    ///
    /// Without the `mlas` feature, MLAS is not linked, so every MLAS-bearing
    /// plan degrades to [`Backend::Native`]. The native path is always present:
    /// it is the correctness baseline every MLAS route is differentially tested
    /// against.
    pub const fn effective(&self) -> Backend {
        if cfg!(feature = "mlas") {
            self.planned
        } else {
            Backend::Native
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
        dtypes: "f32 (f16/bf16 widen to f32 and take the native blocked path)",
        isa: "MLAS runtime dispatch: AVX-512 / AVX2+FMA / SSE2 on x86-64, NEON on aarch64; \
              native SimdX86 needs AVX2+FMA, native Generic needs nothing",
        threads: "MLAS partitions; tiles run on the mlas-sys work-stealing pool under the EP \
                  thread budget",
        shape_gate: "all shapes on x86-64; NXRT_CPU_GEMM_BACKEND=generic|simd|mlas overrides",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::GemmF32,
        planned: Backend::NativeOverMlas,
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
        dtypes: "int4 / int8 weights, f32 activations, f16 scales",
        isa: "native: AVX2/AVX-512/VNNI and NEON dot; MLAS SQNBit: its own dispatch",
        threads: "ours — the persistent SPMD decode pool shards N; MLAS is called per shard",
        shape_gate: "M=1 decode is native (absorbed, #1104); prefill M>=NXRT_SQNBIT_PREFILL_MIN \
                     routes to MLAS SQNBit with a packed-B copy",
        graduation: Graduation::Partial(
            "int4 acc0 decode absorbed at 1.36x/1.56x vs the old borrowed path (#1104); \
             prefill still MLAS",
        ),
    },
    PlanEntry {
        family: KernelFamily::QLinearMatMul,
        planned: Backend::Mlas,
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
        dtypes: "f32",
        isa: "MLAS SIMD exp/reduce dispatch",
        threads: "ours — rows are sharded across the EP pool, MLAS runs a shard serially",
        shape_gate: "contiguous last-axis rows; strided/other-axis cases stay native",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::Normalization,
        planned: Backend::Native,
        dtypes: "f32 / f16 / bf16",
        isa: "native SIMD (AVX2/AVX-512, NEON) with f32 accumulation",
        threads: "ours — rows sharded across the EP pool",
        shape_gate: "all shapes",
        graduation: Graduation::NoMlasPrimitive,
    },
    PlanEntry {
        family: KernelFamily::AttentionTranspose,
        planned: Backend::NativeOverMlas,
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
        dtypes: "f32 <-> u8/i8/int4",
        isa: "MLAS quantize/dequantize SIMD; native block dequant for int4",
        threads: "ours",
        shape_gate: "per-tensor and per-column u8/i8 use MLAS; block/int4 formats are native",
        graduation: Graduation::Partial("block-quantized formats are already native"),
    },
    PlanEntry {
        family: KernelFamily::MoE,
        planned: Backend::NativeOverMlas,
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
        dtypes: "f32",
        isa: "MLAS pooling kernels, including the NCHWc blocked variants",
        threads: "MLAS partitions",
        shape_gate: "MaxPool/AveragePool in f32; other dtypes stay native",
        graduation: Graduation::MlasBaseline,
    },
    PlanEntry {
        family: KernelFamily::Elementwise,
        planned: Backend::Native,
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

static RECORDING: AtomicBool = AtomicBool::new(false);
static INIT: OnceLock<()> = OnceLock::new();

fn log() -> &'static Mutex<Vec<Observation>> {
    static LOG: OnceLock<Mutex<Vec<Observation>>> = OnceLock::new();
    LOG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Honour `NXRT_CPU_DISPATCH_LEDGER=1` exactly once per process.
fn init_from_env() {
    INIT.get_or_init(|| {
        if std::env::var(LEDGER_ENV).as_deref() == Ok("1") {
            RECORDING.store(true, Ordering::Relaxed);
        }
    });
}

/// Whether live recording is on.
#[inline]
pub fn is_recording() -> bool {
    init_from_env();
    RECORDING.load(Ordering::Relaxed)
}

/// Turn live recording on. Intended for tests and benchmarks.
pub fn enable() {
    init_from_env();
    RECORDING.store(true, Ordering::Relaxed);
}

/// Turn live recording off.
pub fn disable() {
    init_from_env();
    RECORDING.store(false, Ordering::Relaxed);
}

/// Drop every recorded observation.
pub fn reset() {
    if let Ok(mut entries) = log().lock() {
        entries.clear();
    }
}

/// Record a routing decision.
///
/// Costs one relaxed atomic load when recording is off, which is the production
/// configuration. Call this at the point the route is *chosen*, not inside the
/// inner loop.
#[inline]
pub fn record(observation: Observation) {
    if !is_recording() {
        return;
    }
    if let Ok(mut entries) = log().lock() {
        entries.push(observation);
    }
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
    #[test]
    fn effective_route_degrades_without_mlas() {
        for entry in PLAN {
            if mlas_linked() {
                assert_eq!(entry.effective(), entry.planned);
            } else {
                assert_eq!(
                    entry.effective(),
                    Backend::Native,
                    "{} must degrade to native without the mlas feature",
                    entry.family
                );
            }
        }
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

    #[test]
    fn recording_is_off_until_asked_for_and_observations_round_trip() {
        // One test, because the recorder is process-global: two tests toggling
        // it in parallel would race each other rather than the code.
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
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].shape, (4, 8, 16));
        assert_eq!(seen[0].isa, Isa::host());
        assert!(observed(KernelFamily::MatMulF32, Backend::Native));
        assert!(!observed(KernelFamily::Softmax, Backend::Native));
        reset();
        disable();
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
}
