//! CPU execution provider exported as an ORT plugin-EP cdylib.
//!
//! This crate produces `libonnx_runtime_ep_cpu_plugin.so` (or platform
//! equivalent) that upstream ONNX Runtime can load via `dlopen` and use as a
//! real execution provider.
//!
//! The crate is intentionally thin: construct the EP, derive kernel-registry
//! entries from the real CPU registry, and export via the C ABI.

use onnx_runtime_ep_cpu::{CpuExecutionProvider, build_cpu_registry_with_descriptors};
use onnx_runtime_ep_plugin::ep::KernelRegistryEntry;

// Attribute this library's allocations to whichever dispatch phase is open.
//
// It has to live here rather than in the harness: ORT `dlopen`s this cdylib, so
// it has its own global allocator, and an allocator installed in the test
// binary never sees an allocation made inside a `Compute` callback. The harness
// reads the totals back through `nxrt_dispatch_probe_snapshot`.
//
// Off unless the `dispatch_probe` feature is on, which no shipped build sets.
#[cfg(feature = "dispatch_probe")]
#[global_allocator]
static PROBE_ALLOC: onnx_runtime_ep_plugin::dispatch_probe::CountingAllocator<std::alloc::System> =
    onnx_runtime_ep_plugin::dispatch_probe::CountingAllocator::new(std::alloc::System);

/// Build `KernelRegistryEntry` slices from the CPU EP's real registry.
///
/// Each entry's `supported_dtypes` is derived from the kernel's actual dispatch
/// implementation via `supported_dtypes_for_op` — fail closed (f32-only for
/// unknown ops). f16/bf16 are advertised only for ops whose kernels genuinely
/// handle them (Add, Sub, Mul, MatMul, etc.).
fn build_kernel_registry_entries() -> Vec<KernelRegistryEntry> {
    let (_registry, descriptors) = build_cpu_registry_with_descriptors();
    descriptors
        .into_iter()
        .map(|d| {
            // Clamp since_version to i32 range (always fits for ONNX opsets).
            let since = d.since_version as i32;
            KernelRegistryEntry {
                op_type: leak_str(&d.op_type),
                domain: leak_str(&d.domain),
                since_version: since,
                // Cover all future opset versions so ORT matches our kernel
                // regardless of the model's declared opset (e.g. Add@7 must
                // still match a model at opset 21).
                end_version: i32::MAX,
                supported_dtypes: d.supported_dtypes,
                input_dtype_constraints:
                    onnx_runtime_ep_cpu::kernels::input_dtype_constraints_for_op(
                        &d.op_type, &d.domain,
                    ),
                output_dtype_constraints:
                    onnx_runtime_ep_cpu::kernels::output_dtype_constraints_for_op(
                        &d.op_type, &d.domain,
                    ),
            }
        })
        .collect()
}

/// Opt this process out of the persistent SPMD decode pool.
///
/// That pool exists for `onnx-genai-engine`'s native decode loop, which runs a
/// whole decode step inside an SPMD scope so resident workers can take one
/// output-column shard each under a single barrier. Nothing in the plugin path
/// ever enters that scope: ONNX Runtime owns the graph, the schedule and the
/// threads, and calls kernels one node at a time. What the plugin does inherit
/// is the pool's costs -- resident workers competing with ORT's own intra-op
/// pool, and a `MatMulNBits` weight pre-partitioned into one MLAS shard per
/// persistent decode worker (`available_parallelism() / 2` by default -- 16 on
/// the 32-vCPU host below), which then caps an unscoped decode GEMV at that
/// worker count no matter how many threads the host has.
///
/// Measured on the plugin path (32 vCPU AMD EPYC 9V74, MLAS build, int4
/// block-32, K=N=2048, M=1, p50 of 41 runs, each backend alone in its own
/// process): 0.376 ms with the pool built against 0.092 ms without it, against
/// ONNX Runtime's own CPU EP at 0.097 ms -- a 3.9x loss turned into a small
/// win. Prefill (`m > 1`) is unaffected either way: it splits the shards into
/// row blocks, so the shard count is not a thread ceiling there.
///
/// Setting `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` explicitly still wins, so a
/// host that wants the pool can ask for it.
///
/// This flips a default for the whole linked copy of `onnx-runtime-ep-cpu`.
/// Today the cdylib has its own copy, separate from any native engine in the
/// process; a future host that linked both against one copy would have to make
/// this scope-aware rather than process-wide.
pub fn disable_persistent_decode_pool() {
    onnx_runtime_ep_cpu::decode_spmd::set_persistent_decode_pool_default(false);
}

/// Whether this library's process would build the persistent SPMD decode pool,
/// as `1` (it would) or `0` (it would not).
///
/// Exists so a test on the other side of the cdylib boundary can observe the
/// opt-out that [`disable_persistent_decode_pool`] performs. A test binary that
/// merely links `onnx-runtime-ep-cpu` sees *its own* copy of the pool statics,
/// not the ones ONNX Runtime's `dlopen`ed library uses, so only an export can
/// answer this. Querying resolves the pool for the process, which is why this
/// is a test hook and not something the EP itself calls.
///
/// # Safety
///
/// None: no arguments, no pointers. Declared `extern "C"` for the test's
/// `dlsym`.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_persistent_decode_pool_built() -> i32 {
    i32::from(onnx_runtime_ep_cpu::decode_spmd::pools().is_some())
}

/// Leak a string to get a `&'static str` (the entries must live for the EP lifetime).
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// ORT plugin-EP entry point: create EP factories with kernel-registry type
/// constraints for f16/bf16 routing.
///
/// # Why hand-written (not `export_ep_factories!`)
///
/// The macro calls `factory::create_ep_factories`, which does not accept a
/// kernel-registry slice. This shim needs
/// `factory::create_ep_factories_with_registry` to advertise typed dtype
/// constraints to ORT. There is no macro variant for that path yet, so the
/// function is written explicitly here. The signature MUST remain identical
/// to the `CreateEpFactories` arm of `export_ep_factories!` in
/// `onnx-runtime-ep-plugin/src/lib.rs`; if that macro arm changes, update
/// this function in lockstep.
///
/// # Safety
///
/// Called by ORT's plugin loader. All pointer arguments must be valid per the
/// ORT plugin-EP C ABI contract.
///
/// # Panic safety
///
/// Any panic is caught; on panic, `*out_num` is set to `0` and an error
/// `OrtStatus` is returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateEpFactories(
    _registration_name: *const ::std::ffi::c_char,
    api_base: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtApiBase,
    _logger: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtLogger,
    out_factories: *mut *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let out_factories_raw = out_factories;
    let out_num_raw = out_num;
    let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        // Before any kernel can query the pool: this is the first call ONNX
        // Runtime makes into the library.
        disable_persistent_decode_pool();
        let entries = build_kernel_registry_entries();
        unsafe {
            onnx_runtime_ep_plugin::factory::create_ep_factories_with_registry(
                api_base,
                out_factories_raw,
                max_factories,
                out_num_raw,
                || Box::new(CpuExecutionProvider::new()),
                entries,
            )
        }
    }));
    match result {
        Ok(status) => status,
        Err(_panic_payload) => {
            if !out_num_raw.is_null() {
                unsafe { *out_num_raw = 0 };
            }
            onnx_runtime_ep_plugin::panic_to_fail_status(
                "CreateEpFactories: constructor panicked; plugin not loaded (fail-closed)",
            )
        }
    }
}

/// ORT plugin-EP entry point: release an EP factory.
///
/// # ABI reference
///
/// `onnxruntime_ep_c_api.h:2669`:
/// ```c
/// typedef OrtStatus* (*ReleaseEpApiFactoryFn)(_In_ OrtEpFactory* factory);
/// ```
/// Returns `nullptr` on success or a non-null `OrtStatus*` on error/panic.
///
/// # Safety
///
/// `factory` must be a pointer returned by `CreateEpFactories` from this
/// library, and must not be used after this call.
///
/// # Panic safety
///
/// Any panic inside the release path is caught and surfaced as a failure
/// `OrtStatus`. Unwinding into ORT would be undefined behaviour.
///
/// # Kept in sync with `export_ep_factories!`
///
/// This shim stays hand-written because `CreateEpFactories` above calls
/// `create_ep_factories_with_registry` (not available in the macro). The body
/// below is intentionally identical to the `ReleaseEpFactory` arm of the
/// `export_ep_factories!` macro in `onnx-runtime-ep-plugin/src/lib.rs`.
/// If that macro arm changes, this must change in lockstep.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(
    factory: *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        // SAFETY: caller guarantees the pointer was returned by
        // CreateEpFactories from this library.
        unsafe { onnx_runtime_ep_plugin::factory::release_ep_factory(factory) }
    }));
    match result {
        ::std::result::Result::Ok(status) => status,
        ::std::result::Result::Err(_panic_payload) => onnx_runtime_ep_plugin::panic_to_fail_status(
            "ReleaseEpFactory: panic during factory release (fail-closed)",
        ),
    }
}

// ─── Test observability: compiled-node counter ───────────────────────────────

/// Returns the number of nodes compiled by our EP since last reset.
/// Exported as a C symbol so integration tests (which load us via dlopen)
/// can assert that our EP actually claimed and compiled nodes.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_compiled_node_count() -> usize {
    onnx_runtime_ep_plugin::ep::compiled_node_count()
}

/// Number of node inputs this EP reported to its kernels as session-lifetime
/// constants. Read through `dlopen` by the plugin E2E suite to prove weights
/// are recognised as weights on the ORT plugin path, where ORT presents a
/// fused node's initializers as ordinary inputs.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_constant_weight_inputs() -> usize {
    onnx_runtime_ep_plugin::ep::constant_weight_inputs()
}

/// Resets the constant-weight-input counter to zero.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_constant_weight_inputs() {
    onnx_runtime_ep_plugin::ep::reset_constant_weight_inputs()
}

/// Resets the compiled-node counter to zero.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_compiled_node_count() {
    onnx_runtime_ep_plugin::ep::reset_compiled_node_count()
}

/// Number of times ORT entered this EP's C-ABI `GetCapability` callback.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_get_capability_call_count() -> usize {
    onnx_runtime_ep_plugin::ep::get_capability_call_count()
}

/// Resets the C-ABI `GetCapability` callback counter.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_get_capability_call_count() {
    onnx_runtime_ep_plugin::ep::reset_get_capability_call_count()
}

/// Number of node kernels this EP has **executed** since the last reset.
///
/// The compiled-node counter above says ORT assigned us the node; this one says
/// our kernel is what ran when the session was executed. The no-defer rule
/// needs both: assignment without execution is not ownership, and an output
/// that matches ORT proves nothing about which EP produced it.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_executed_node_count() -> usize {
    onnx_runtime_ep_plugin::compute::executed_node_count()
}

/// Resets the executed-node counter to zero.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_executed_node_count() {
    onnx_runtime_ep_plugin::compute::reset_executed_node_count()
}

/// Enable and reset route-scoped CPU Einsum concurrency observation.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_einsum_concurrency_probe() {
    onnx_runtime_ep_cpu::kernels::einsum::reset_concurrency_probe()
}

/// Disable observation and return the maximum overlap for `route`.
///
/// Routes are `0=view-copy`, `1=reduction/oracle`,
/// `2=materialized-GEMM`, and `3=generic/tree`. An unknown route returns zero.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_finish_einsum_concurrency_probe(route: usize) -> usize {
    onnx_runtime_ep_cpu::kernels::einsum::finish_concurrency_probe()
        .get(route)
        .copied()
        .unwrap_or(0)
}

/// Reset successful CPU Einsum route counters.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_einsum_route_telemetry() {
    onnx_runtime_ep_cpu::kernels::einsum::reset_route_telemetry()
}

/// Successful CPU Einsum dispatches for one route index.
///
/// Indices are documented by
/// `onnx_runtime_ep_cpu::kernels::einsum::route_telemetry_count`.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_einsum_route_count(route: usize) -> usize {
    onnx_runtime_ep_cpu::kernels::einsum::route_telemetry_count(route)
}

// ─── Build identity ─────────────────────────────────────────────────────────

/// The optional build features compiled into this cdylib, as a NUL-terminated
/// static string: `"mlas"`, or empty when there are none.
///
/// A packaged cdylib is opaque: nothing about the file says whether the
/// vendored MLAS kernels were linked in, and the difference is an order of
/// magnitude on the quantized matmul paths. Exporting it lets the wheel's own
/// smoke test assert that what shipped is what was intended, instead of
/// discovering a pure-Rust build in production benchmarks.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_build_features() -> *const ::std::os::raw::c_char {
    BUILD_FEATURES.as_ptr() as *const ::std::os::raw::c_char
}

#[cfg(feature = "mlas")]
const BUILD_FEATURES: &[u8] = b"mlas\0";
#[cfg(not(feature = "mlas"))]
const BUILD_FEATURES: &[u8] = b"\0";

/// Number of workspace **placement resolutions** since the last reset.
///
/// Mirrors the CUDA plugin's accessor so the same validation harness runs
/// against either cdylib: on a GPU-less host it can be pointed at this library
/// to exercise its own plumbing before the GPU host runs it for real.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_workspace_placement_queries() -> usize {
    onnx_runtime_ep_plugin::compute::workspace_placement_queries()
}

/// Resets the workspace placement-resolution counter to zero.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_workspace_placement_queries() {
    onnx_runtime_ep_plugin::compute::reset_workspace_placement_queries()
}
