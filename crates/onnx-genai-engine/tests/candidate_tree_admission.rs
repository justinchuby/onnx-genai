use onnx_genai_engine::{
    DeviceCompatibilityDomain, DeviceMemoryAuthority, Engine, EngineConfig,
    MemoryAuthorityProvider, ProcessMemoryManager, ResourceLimit,
};
use onnx_genai_ort::SessionOptions;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct NeverCalledAuthorityProvider;

impl MemoryAuthorityProvider for NeverCalledAuthorityProvider {
    fn process_memory_manager(&self) -> ProcessMemoryManager {
        panic!("candidate-tree admission must precede memory-authority construction")
    }

    fn validate_limit(
        &self,
        _domain: &DeviceCompatibilityDomain,
        _requested: ResourceLimit,
    ) -> anyhow::Result<()> {
        panic!("candidate-tree admission must precede memory-authority validation")
    }

    fn authority(
        &self,
        _domain: &DeviceCompatibilityDomain,
        _resolved_limit_bytes: u64,
    ) -> anyhow::Result<DeviceMemoryAuthority> {
        panic!("candidate-tree admission must precede memory-authority construction")
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/unsupported-candidate-tree")
}

fn authority_provider() -> Arc<dyn MemoryAuthorityProvider> {
    Arc::new(NeverCalledAuthorityProvider)
}

fn staged_invalid_fixture(name: &str, from: &str, to: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/candidate-tree-admission")
        .join(name);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create invalid candidate-tree fixture");
    let source = fs::read_to_string(fixture().join("inference_metadata.yaml"))
        .expect("read candidate-tree fixture");
    assert!(source.contains(from), "fixture no longer contains {from:?}");
    fs::write(
        root.join("inference_metadata.yaml"),
        source.replace(from, to),
    )
    .expect("write invalid candidate-tree fixture");
    root
}

fn staged_missing_component_fixture(name: &str) -> PathBuf {
    staged_invalid_fixture(
        name,
        "implementation: {kind: binding}",
        "implementation: {kind: onnx, artifact: must-not-load.onnx}",
    )
}

fn assert_candidate_tree_refusal(result: anyhow::Result<()>, constructor: &str) {
    let error = result.expect_err("candidate-tree package must fail closed");
    let message = format!("{error:#}");
    assert!(
        message.contains("candidate-tree")
            && message.contains("onnx-genai.speculative@1")
            && message.contains("no candidate-tree package-dispatch capability or executor"),
        "{constructor} did not report the exact unsupported contract and capability: {message}"
    );
    assert!(
        message.contains("Refusing to silently run plain or MTP generation"),
        "{constructor} did not explain the fail-closed dispatch decision: {message}"
    );
    assert!(
        !message.contains("must-not-load.onnx"),
        "{constructor} resolved or inspected a component artifact before runtime admission: \
         {message}"
    );
}

#[test]
fn candidate_tree_engine_from_dir_fails_before_component_loading() {
    let fixture = staged_missing_component_fixture("engine-from-dir");
    assert_candidate_tree_refusal(
        Engine::from_dir(&fixture, EngineConfig::default()).map(|_| ()),
        "Engine::from_dir",
    );
}

#[test]
fn candidate_tree_engine_session_options_fails_before_component_loading() {
    let fixture = staged_missing_component_fixture("engine-session-options");
    assert_candidate_tree_refusal(
        Engine::from_dir_with_session_options(
            &fixture,
            EngineConfig::default(),
            SessionOptions::default(),
        )
        .map(|_| ()),
        "Engine::from_dir_with_session_options",
    );
}

#[test]
fn candidate_tree_engine_memory_authority_fails_before_mutation() {
    let fixture = staged_missing_component_fixture("engine-memory-authority");
    assert_candidate_tree_refusal(
        Engine::from_dir_with_memory_authority_provider(
            &fixture,
            EngineConfig::default(),
            authority_provider(),
        )
        .map(|_| ()),
        "Engine::from_dir_with_memory_authority_provider",
    );
}

#[test]
fn candidate_tree_semantic_errors_precede_runtime_capability_admission() {
    for (fixture, expected) in [
        (
            staged_invalid_fixture(
                "invalid-topology",
                "output: candidate_parents",
                "output: absent_topology",
            ),
            "absent_topology",
        ),
        (
            staged_invalid_fixture("invalid-version", "version: '1'", "version: '2'"),
            "version",
        ),
    ] {
        let Err(error) = Engine::from_dir(&fixture, EngineConfig::default()) else {
            panic!("invalid speculative metadata must fail before runtime admission");
        };
        let message = format!("{error:#}");
        assert!(
            message.contains("Invalid inference metadata") && message.contains(expected),
            "semantic validation did not identify {expected}: {message}"
        );
        assert!(
            !message.contains("no candidate-tree package-dispatch capability or executor"),
            "runtime capability refusal masked a semantic metadata error: {message}"
        );
    }
}
