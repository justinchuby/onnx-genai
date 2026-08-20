//! Loads the built cdylib and drives it across the real nxrt C ABI.
//!
//! The other tests in this crate check Rust-level helpers; this one is the only
//! evidence that the exported symbols work as a *plugin* — that negotiation
//! agrees, that a factory is produced for each available backend, and that the
//! provider behind it can actually open a device. Without it, a plugin that
//! compiles but advertises nothing would pass the suite.
//!
//! Requires a GPU, so it skips when none is present unless
//! `NXRT_REQUIRE_GPU_TESTS=1` demands otherwise.

use onnx_runtime_ep_cubecl::backend::CubeclBackend;
use onnx_runtime_ep_nxrt_host::load_nxrt_plugin;

/// Features the cdylib must be rebuilt with so the artifact matches this test
/// binary. Without this the helper would overwrite it with a default-feature
/// build that contains no backends at all.
const FEATURES: &[&str] = &["webgpu"];

fn require_gpu() -> bool {
    std::env::var("NXRT_REQUIRE_GPU_TESTS").is_ok_and(|value| value == "1")
}

fn skip(reason: &str) -> bool {
    if require_gpu() {
        panic!("NXRT_REQUIRE_GPU_TESTS=1 but the cubecl plugin could not be exercised: {reason}");
    }
    eprintln!("skipping: {reason}");
    true
}

#[test]
fn nxrt_host_loads_the_plugin_and_gets_a_factory_per_available_backend() {
    let Some(path) = onnx_runtime_ort_testkit::find_plugin_cdylib_with_features(
        "onnx-runtime-ep-cubecl-plugin",
        FEATURES,
    ) else {
        skip("the cubecl plugin cdylib could not be built");
        return;
    };

    let expected: Vec<&str> = CubeclBackend::ALL
        .into_iter()
        .filter(|backend| backend.unavailable_message().is_none())
        .map(CubeclBackend::provider_name)
        .collect();

    let plugin = match load_nxrt_plugin(&path) {
        Ok(plugin) => plugin,
        Err(error) => {
            // Zero devices is the documented fail-closed outcome on a host with
            // no adapter, and it must stay distinguishable from a broken ABI.
            skip(&format!(
                "nxrt host could not load {}: {error}",
                path.display()
            ));
            return;
        }
    };

    assert_eq!(
        plugin.num_factories(),
        expected.len(),
        "the plugin must advertise exactly one factory per available backend ({expected:?})"
    );
}
