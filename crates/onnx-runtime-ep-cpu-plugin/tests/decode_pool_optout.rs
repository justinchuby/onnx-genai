//! The plugin EP must not build the persistent SPMD decode pool.
//!
//! Its own test binary: [`onnx_runtime_ep_cpu::decode_spmd::pools`] resolves
//! once per process, so a sibling test that ran a kernel first would fix the
//! answer before this one could observe it.

/// `CreateEpFactories` opts the process out, and the pool then never builds.
///
/// Nothing in the plugin path enters an SPMD decode scope, so a built pool is
/// pure cost: resident workers competing with ONNX Runtime's intra-op pool, and
/// a `MatMulNBits` weight pre-split into one MLAS shard per persistent decode
/// worker, which caps an unscoped decode GEMV at that worker count. Measured
/// 0.376 ms -> 0.092 ms on int4 block-32 K=N=2048 M=1 (ORT's CPU EP: 0.097 ms).
///
/// This covers the library plumbing only. The falsifier for the *call site* in
/// `CreateEpFactories` is `the_plugin_ep_disables_the_decode_pool_in_ort` in
/// `plugin_ort_e2e.rs`, which reads the answer back across the cdylib boundary
/// after ONNX Runtime has loaded and used the library.
#[test]
fn the_plugin_ep_does_not_build_the_persistent_decode_pool() {
    assert!(
        std::env::var("ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL").is_err(),
        "this test describes the unset default; the environment overrides it"
    );

    onnx_runtime_ep_cpu_plugin::disable_persistent_decode_pool();

    assert!(
        onnx_runtime_ep_cpu::decode_spmd::pools().is_none(),
        "the plugin EP built the persistent SPMD decode pool it never dispatches to"
    );
}
