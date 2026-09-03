//! # `onnx-runtime-ep-cpu`
//!
//! The CPU execution provider for the ORT 2.0 runtime (see `docs/architecture/ORT2.md` §4.4
//! and §54 Phase 1). It implements [`onnx_runtime_ep_api::ExecutionProvider`]
//! and hosts pure-Rust reference kernels for the Phase-1 op set (`MatMul`,
//! `Add`, `Relu`, `Reshape`, `Transpose`, `Gather`, `LayerNormalization`).
//!
//! ## Backends: correctness baseline + SIMD fast path
//!
//! The GEMM hot spot is served through [`backend::CpuBackend`] (`docs/architecture/ORT2.md`
//! §25.2). The **default** backend is a pure-Rust blocked, register-tiled,
//! rayon-parallelized f32 GEMM — the portable, offline correctness baseline that
//! compiles anywhere with no C++/FFI. On supported x86 hosts, the built-in
//! `SimdX86` implementation provides the default fast path. Every backend lives behind the
//! [`onnx_runtime_ep_api::Kernel`] trait, so neither the EP contract nor the
//! session observes which one ran. See [`kernels::matmul`] for the hot spot.
//!
//! ## `unsafe`
//!
//! The default (Generic) path is `unsafe`-minimal: the only `unsafe` is the raw
//! device-buffer access the ep-api contract forces (aligned host
//! `alloc`/`dealloc`, `memcpy`, and strided element reads/writes), each isolated
//! and `SAFETY`-documented. The blocked rayon GEMM itself contains no `unsafe`;
//! all kernel arithmetic is safe Rust operating on
//! dense `Vec<f32>` buffers produced by the two audited accessors in [`kernels`].

// Kernel entry points mirror ONNX operator schemas, whose independent tensors and
// dimensions often exceed Clippy's generic argument-count threshold.
#![allow(clippy::too_many_arguments)]

pub mod backend;
pub mod backend_ab;
pub mod core_topology;
pub mod decode_affinity;
pub mod decode_numa;
pub mod decode_spmd;
pub mod dispatch_ledger;
pub mod dtype;
pub mod kernels;
#[cfg(all(feature = "mlas", feature = "ops-cnn"))]
pub mod nchwc_layout;
pub mod optimizer;
pub(crate) mod persistent_pool_width;
pub mod provider;
pub mod strided;
pub mod task_runtime;
#[cfg(test)]
pub(crate) mod test_support;
mod trace;
pub mod weight_offload;

pub use backend::CpuBackend;
pub use dispatch_ledger::{Backend as DispatchBackend, KernelFamily, effective_backend};
pub use kernels::qmoe::WeightOffloadHostCache;
pub use kernels::{CpuOpDescriptor, build_cpu_registry_with_descriptors, supported_dtypes_for_op};
pub use optimizer::{
    ConvBatchNormActivationFusion, MatMulNBitsBiasFusion, ProjectionFusion, SiblingProjectionMerge,
    cpu_optimization_passes,
};
pub use provider::CpuExecutionProvider;
pub use weight_offload::placement::{
    GpuLayersOverrideReport, HostFallbackReason, IqFormat, LayerPlacement, LayerWeightRegions,
    Placement, PlacementError, PlacementPlan, QuantTileFormat, RegionPlacement, SnappedTileSize,
    TileSizeError, plan_placement, snap_transfer_tile_bytes,
};
pub use weight_offload::weight_handle::{
    ExecutionProviderCapabilities, LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary,
    NXRT_WEIGHT_PAGING_CAPABILITY, NegotiatedWeight, Phase3aHostOnlyBinder, ResidentWeight,
    ResidentWeightMaterializer, WeightHandle, WeightHandleError,
};
pub use weight_offload::{
    LinuxProcessMemoryStats, WEIGHT_OFFLOAD_ENV, WEIGHT_OFFLOAD_HOST_BYTES_ENV,
    WeightOffloadLayerStats, WeightOffloadStats, weight_offload_stats,
};

pub use kernels::selection::non_max_suppression;
pub use kernels::slice::{SliceAxisPlan, slice_axes_steps, slice_plan};

// Einsum Float32 workspace is parked per execution thread, so a per-buffer cap
// alone would multiply by the pool width. Each provider/session owns an
// immutable retention verdict; all admitted sessions share only the governed
// process byte ceiling and live-byte accounting.
pub use kernels::einsum::{
    EinsumScratchRetention, einsum_scratch_budget_predicted_bytes, einsum_scratch_live_bytes,
    einsum_scratch_process_cap_bytes,
};
pub use kernels::matmul_nbits::bound_process_to_decode_budget;
pub use kernels::matmul_nbits::set_decode_thread_budget;
pub use kernels::matmul_nbits::with_decode_pool_scope;
pub use kernels::matmul_nbits::{
    matmul_nbits_decode_caches_dequant_f32, matmul_nbits_resident_side_cache_bytes,
    mlas_sqnbit_packed_live_bytes, resident_dequant_f32_cache_bytes,
    set_mlas_sqnbit_packing_enabled, set_resident_dequant_f32_cache_enabled,
};
// #1056: a resident, session-lifetime, weight-scaled buffer must be reportable
// in bytes. The entry-count accessor stays for the benchmarks that assert reuse.
pub use kernels::group_query_attention::{present_inplace_count, present_inplace_half_count};
pub use kernels::matmul::weight_transpose_cache_bytes;
// #1056: the transpose cache is now *governed*, not just reported: the plan
// budgets `weight_transpose_cache_predicted_bytes` and, when it does not fit,
// declines it via `set_weight_transpose_cache_enabled` (kernels then transpose
// per call and retain nothing).
pub use kernels::matmul::{
    set_weight_transpose_cache_enabled, weight_transpose_cache_predicted_bytes,
};
// #1056: the per-kernel `MatMulPrepack::dense` widened-f32 cache is the fourth
// resident, weight-scaled buffer brought under the plan. The plan budgets
// `matmul_dense_cache_predicted_bytes` and, when it does not fit, declines it via
// `set_matmul_dense_cache_enabled` (kernels then widen per call and retain
// nothing, byte-identical output).
pub use kernels::matmul::{matmul_dense_cache_predicted_bytes, set_matmul_dense_cache_enabled};
// #1133: the QLinearMatMul per-thread `i32` accumulator scratch was bounded by a
// per-buffer constant but parked on every worker thread, so the real ceiling was
// `32 MiB x threads`. It is now a process-wide, declinable budget: the plan
// budgets `qlinear_accumulator_budget_predicted_bytes` and declines it via
// `set_qlinear_accumulator_budget_admitted` (kernels then reallocate per call,
// byte-identical). The constant-`B` MLAS pre-pack (`packed_b`, ungoverned since
// `ac394fd6`) is the second buffer in the same kernel brought under the plan.
pub use kernels::qlinear_matmul::{
    qlinear_accumulator_budget_predicted_bytes, qlinear_accumulator_live_bytes,
    qlinear_accumulator_process_cap_bytes, qlinear_packed_b_live_bytes,
    qlinear_packed_b_predicted_bytes, set_qlinear_accumulator_budget_admitted,
    set_qlinear_accumulator_process_cap_bytes, set_qlinear_packed_b_enabled,
};

#[cfg(test)]
mod feature_default_guard {
    /// `mlas` must stay off by default.
    ///
    /// The vendored MLAS kernels are a research/reference arm, not what ships.
    /// This matters beyond a build flag: `bench_generic` used to *require* the
    /// feature, so every published A/B ratio measured the reference arm while
    /// being read as a production number, and the production pure-Rust softmax
    /// turned out to be about 9x ORT rather than the ~1.0 the tables showed.
    /// If a default build ever links MLAS again, the same class of mistake
    /// becomes possible again, so fail loudly here.
    ///
    /// Resolved from the **manifest**, not from `cfg!`. Cargo features are
    /// additive and indistinguishable at `cfg` time, so `!cfg!(feature =
    /// "mlas")` cannot tell "`mlas` became a default" from "`mlas` was
    /// requested on the command line": it passed in the shipped configuration
    /// (where nothing could have gone wrong yet) and *failed* in the research
    /// one (where nothing is wrong at all), which is the opposite of what this
    /// guard is for. The manifest answers the actual policy question in every
    /// configuration, including `--features mlas`.
    ///
    /// `crates/onnx-runtime-ep-cpu-plugin/tests/default_artifacts_are_mlas_free.rs`
    /// makes the same statement about the shipped cdylib, its resolved
    /// dependency graph and the wheel. This one is here so that editing *this*
    /// crate's `Cargo.toml` and running `cargo test -p onnx-runtime-ep-cpu`
    /// is enough to catch it.
    #[test]
    fn mlas_is_not_a_default_feature() {
        const MANIFEST: &str = include_str!("../Cargo.toml");

        let default = feature_list(MANIFEST, "default");
        assert!(
            !default.is_empty(),
            "probe read no `default` features, so the assertions below would \
             pass vacuously"
        );
        // Transitive: `default` contains the `full` umbrella, so a `full` that
        // gained MLAS would activate it for every consumer without the word
        // `default` appearing anywhere near the change.
        let mut reachable = default.clone();
        let mut frontier = default;
        while let Some(feature) = frontier.pop() {
            for implied in feature_list(MANIFEST, &feature) {
                if !reachable.contains(&implied) {
                    reachable.push(implied.clone());
                    frontier.push(implied);
                }
            }
        }
        // Substring, not `== "mlas"`: `dep:mlas-sys` and
        // `onnx-runtime-ep-cpu/mlas` reach the reference just as well, and
        // nothing else in this workspace is named after MLAS.
        let offenders: Vec<&String> = reachable
            .iter()
            .filter(|f| f.to_ascii_lowercase().contains("mlas"))
            .collect();
        assert!(
            offenders.is_empty(),
            "the default feature closure of onnx-runtime-ep-cpu reaches the \
             MLAS reference through {offenders:?}: production would then ship \
             MLAS and every benchmark arm label would be wrong"
        );
    }

    /// Members of a `name = [...]` feature list, or empty when there is none.
    ///
    /// Deliberately a five-line scan rather than a TOML dependency: this runs
    /// against one hand-written file whose shape is fixed by
    /// `feature_lists_are_read_not_guessed`.
    fn feature_list(manifest: &str, name: &str) -> Vec<String> {
        // Find `name = [` at the start of a line, tolerating any spacing around
        // the `=`. Anchoring on one exact spelling would let a reformat turn a
        // populated feature into a silently empty one, which is the direction
        // this guard must never fail in.
        let start = manifest.lines().enumerate().find_map(|(i, line)| {
            let rest = line.strip_prefix(name)?.trim_start();
            let rest = rest.strip_prefix('=')?.trim_start();
            rest.starts_with('[')
                .then(|| manifest.lines().take(i).map(|l| l.len() + 1).sum::<usize>())
        });
        let Some(start) = start else {
            return Vec::new();
        };
        let open = manifest[start..].find('[').expect("array opens") + start;
        let close = manifest[open..].find(']').expect("array closes") + open;
        manifest[open + 1..close]
            .split(',')
            .map(|item| item.trim().trim_matches('"').trim().to_string())
            .filter(|item| !item.is_empty() && !item.starts_with('#'))
            .collect()
    }

    /// The guard above is a filter over text; a filter that reads nothing
    /// passes every assertion built on it. This is its positive control.
    #[test]
    fn feature_lists_are_read_not_guessed() {
        const MANIFEST: &str = include_str!("../Cargo.toml");
        assert!(
            feature_list(MANIFEST, "default").contains(&"full".to_string()),
            "the manifest reader must see `default = [\"full\"]`"
        );
        assert!(
            feature_list(MANIFEST, "full").contains(&"ops-core".to_string()),
            "the manifest reader must see through the multi-line `full` umbrella"
        );
        assert!(
            feature_list(MANIFEST, "mlas").contains(&"dep:mlas-sys".to_string()),
            "the manifest still has to declare the opt-in research feature"
        );
        assert!(
            feature_list(MANIFEST, "no-such-feature").is_empty(),
            "an absent feature must read as empty, not panic"
        );

        // The transitive walk must actually catch MLAS reaching `default`
        // through the umbrella, which is the regression this guard exists for.
        let synthetic =
            "\ndefault = [\"full\"]\nfull = [\"ops-core\", \"mlas\"]\nmlas = [\"dep:mlas-sys\"]\n";
        assert!(
            feature_list(synthetic, "full").iter().any(|f| f == "mlas"),
            "a `full` umbrella that gained MLAS must be visible to the reader"
        );

        // Spacing must not turn a populated feature into an empty one: an
        // empty read is a *pass* for every caller, so the reader has to be
        // insensitive to formatting the manifest is free to change.
        for spelling in [
            "full=[\"mlas\"]",
            "full  =  [\"mlas\"]",
            "full\t= [\"mlas\"]",
        ] {
            let manifest = format!("\n[features]\n{spelling}\n");
            assert_eq!(
                feature_list(&manifest, "full"),
                vec!["mlas".to_string()],
                "`{spelling}` must read the same as the canonical spelling"
            );
        }

        // A feature name must not match as a suffix or a prefix of another.
        let neighbours = "\nnot-full = [\"mlas\"]\nfull-extra = [\"mlas\"]\n";
        assert!(
            feature_list(neighbours, "full").is_empty(),
            "`full` must not be found inside `not-full` or `full-extra`"
        );
    }
}
