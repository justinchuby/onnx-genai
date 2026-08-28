//! Runtime admission coverage for structurally valid but unimplemented DFlash.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_engine::{
    DeviceCompatibilityDomain, DeviceMemoryAuthority, Engine, EngineConfig,
    MemoryAuthorityProvider, PackageCapabilityError, ResourceLimit, package_capability_error,
};
use onnx_genai_ort::SessionOptions;
use onnx_runtime_memory_governor::ProcessMemoryManager;

struct PanicAuthorityProvider;

impl MemoryAuthorityProvider for PanicAuthorityProvider {
    fn process_memory_manager(&self) -> ProcessMemoryManager {
        panic!("DFlash admission must precede process memory-manager allocation")
    }

    fn validate_limit(
        &self,
        _domain: &DeviceCompatibilityDomain,
        _requested: ResourceLimit,
    ) -> anyhow::Result<()> {
        panic!("DFlash admission must precede authority validation")
    }

    fn authority(
        &self,
        _domain: &DeviceCompatibilityDomain,
        _resolved_limit_bytes: u64,
    ) -> anyhow::Result<DeviceMemoryAuthority> {
        panic!("DFlash admission must precede device-authority allocation")
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dflash-admission")
}

fn copied_fixture(name: &str) -> anyhow::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/dflash-admission")
        .join(format!("{name}-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir_all(&root)?;
    for file in [
        "inference_metadata.yaml",
        "target.onnx.textproto",
        "proposer.onnx.textproto",
    ] {
        fs::copy(fixture().join(file), root.join(file))?;
    }
    Ok(root)
}

fn assert_dflash_refusal(error: anyhow::Error) {
    let capability = package_capability_error(&error).expect("DFlash refusal remains typed");
    assert!(matches!(
        capability,
        PackageCapabilityError::DFlashExecutionUnavailable {
            ref version,
            ref capability,
        } if version == "1"
            && capability == onnx_genai_metadata::capabilities::DFLASH_FLAT_BLOCK
    ));
    let message = format!("{error:#}");
    assert!(
        message.contains("onnx-genai.dflash-flat-block@1")
            && message.contains("before model/session allocation")
            && message.contains("output-family handling"),
        "{message}"
    );
}

#[test]
fn every_public_engine_constructor_refuses_before_model_session_or_authority_allocation() {
    let root = fixture();
    assert!(
        onnx_genai_ort::PipelineModelDirectory::load(&root).is_err(),
        "the empty ONNX files are a loader spy: reaching model admission must fail"
    );

    assert_dflash_refusal(
        Engine::from_dir(&root, EngineConfig::default())
            .err()
            .expect("Engine::from_dir must refuse DFlash"),
    );
    assert_dflash_refusal(
        Engine::from_dir_with_session_options(
            &root,
            EngineConfig::default(),
            SessionOptions::default(),
        )
        .err()
        .expect("Engine::from_dir_with_session_options must refuse DFlash"),
    );
    assert_dflash_refusal(
        Engine::from_dir_with_memory_authority_provider(
            &root,
            EngineConfig::default(),
            Arc::new(PanicAuthorityProvider),
        )
        .err()
        .expect("Engine::from_dir_with_memory_authority_provider must refuse DFlash"),
    );
}

#[test]
fn unknown_dflash_version_keeps_the_specific_validation_diagnostic() -> anyhow::Result<()> {
    let root = copied_fixture("unknown-version")?;
    let metadata_path = root.join("inference_metadata.yaml");
    let metadata =
        fs::read_to_string(&metadata_path)?.replace("    version: \"1\"", "    version: \"99\"");
    fs::write(metadata_path, metadata)?;

    let error = Engine::from_dir(&root, EngineConfig::default())
        .err()
        .expect("unknown DFlash versions fail validation");
    let message = format!("{error:#}");
    assert!(
        message.contains("unsupported DFlash flat-block contract version '99'")
            && message.contains("supported versions are '1' (base)")
            && message.contains("'2' (selector_convolution_v1)"),
        "{message}"
    );
    assert!(
        package_capability_error(&error).is_none(),
        "malformed contracts must fail validation before execution capability admission"
    );
    Ok(())
}

#[test]
fn plain_workflow_construction_remains_unaffected() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let engine = Engine::from_dir(&root, EngineConfig::default())?;
    assert!(engine.dflash_diagnostic().is_none());
    Ok(())
}

#[test]
fn dflash_execution_and_manual_commit_helpers_consume_the_canonical_guard() {
    let engine_api = include_str!("../src/engine/workflow_api.rs");
    let runtime_api = include_str!("../src/pipeline/speculative.rs");
    let guarded_body = |source: &str, helper: &str, guard: &str| {
        let start = source
            .find(&format!("pub fn {helper}"))
            .unwrap_or_else(|| panic!("{helper} declaration is present"));
        let tail = &source[start..];
        let end = tail[1..]
            .find("\n    pub fn ")
            .map_or(tail.len(), |end| end + 1);
        let body = &tail[tail
            .find('{')
            .unwrap_or_else(|| panic!("{helper} has a function body"))
            + 1..end];
        assert!(
            body.trim_start().starts_with(guard),
            "{helper} must consume {guard} as its first statement before inspecting or mutating \
             DFlash values"
        );
    };
    for helper in [
        "propose_dflash",
        "verify_dflash",
        "begin_dflash_state_transaction",
        "commit_dflash_state_transaction",
        "abort_dflash_state_transaction",
    ] {
        guarded_body(
            engine_api,
            helper,
            "self.require_workflow_execution_admitted()?",
        );
        guarded_body(runtime_api, helper, "self.require_execution_admitted()?");
    }
}
