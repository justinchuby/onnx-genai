#![cfg(feature = "native-backend")]
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateConstraint, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice, ResourceLimit, ResourceLimits,
};
use std::path::Path;
mod decoder {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/build_literal_decoder.rs"
    ));
}
fn request(limit: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1]));
    request.options.max_new_tokens = 1;
    request.options.greedy = true;
    request.options.temperature = 0.;
    request.options.max_constraint_bytes = limit;
    request
        .options
        .logit_bias
        .insert(22, onnx_genai_engine::logits::LogitBias::Add(1000.));
    request
        .options
        .logit_bias
        .insert(23, onnx_genai_engine::logits::LogitBias::Add(1001.));
    request.options.constraint = Some(GenerateConstraint::JsonSchema {
        schema: "{\"const\":7}".into(),
        prefix: String::new(),
    });
    request
}
#[test]
fn native_literal_decoder_governor_denies_before_generation_and_independent_retry_recovers() {
    let root = tempfile::tempdir().unwrap();
    decoder::build(
        root.path(),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
        false,
    );
    let mut engine = Engine::from_dir(
        root.path(),
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            limits: ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(160 * 1024 * 1024),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let baseline = engine.resource_snapshot().host_ram.used;
    let mut admissions = 0;
    let mut admitted = || {
        admissions += 1;
    };
    let error = engine
        .generate_with_callbacks(request(8 * 1024 * 1024), Some(&mut admitted), None)
        .unwrap_err();
    assert_eq!(
        admissions, 0,
        "workspace must deny before admission/execution"
    );
    assert!(
        format!("{error:#}").contains("bounded contextual literal decoder before generation"),
        "{error:#}"
    );
    assert_eq!(
        engine.resource_snapshot().host_ram.used,
        baseline,
        "failed workspace admission cannot retain a lease"
    );
    let mut invalid = request(4096);
    invalid.options.constraint = Some(GenerateConstraint::JsonSchema {
        schema: r#"{"type":"invalid-schema-type"}"#.into(),
        prefix: String::new(),
    });
    let error = engine.generate(invalid).unwrap_err();
    assert!(
        error.is::<onnx_genai_engine::ProcessorConfigurationError>(),
        "{error:#}"
    );
    assert_eq!(
        engine.resource_snapshot().host_ram.used,
        baseline,
        "failed plan compilation releases its admitted workspace"
    );
    for _ in 0..2 {
        let result = engine.generate(request(4096)).unwrap();
        assert_eq!(result.text, "7");
        assert_eq!(result.token_ids, vec![22]);
    }
}

#[test]
fn native_literal_byte_group_cancellation_discards_pending_state_and_retries() {
    for byte_level in [false, true] {
        let root = tempfile::tempdir().unwrap();
        decoder::invalid_group(
            root.path(),
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
            byte_level,
        );
        let mut engine = Engine::from_dir(
            root.path(),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..Default::default()
            },
        )
        .unwrap();
        let mut request = request(4096);
        request.options.max_new_tokens = 3;
        request.options.logit_bias.clear();
        let control = onnx_genai_engine::GenerationControl::new();
        let cancel = control.clone();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = seen.clone();
        let mut callback = move |token: onnx_genai_engine::GenerateToken| {
            assert!(
                token.text.is_empty(),
                "incomplete replacement prefix must remain unpublished"
            );
            let mut seen = observed.lock().unwrap();
            seen.push(token.token_id);
            if seen.len() == 2 {
                cancel.cancel();
            }
            Ok(())
        };
        let error = engine
            .generate_with_control_callbacks(request.clone(), control, None, Some(&mut callback))
            .expect_err("cancelled provisional generation aborts its Native transaction");
        assert!(
            error.is::<onnx_genai_engine::GenerationCancelled>(),
            "{error:#}"
        );
        assert_eq!(*seen.lock().unwrap(), vec![22, 23]);
        let retry = engine.generate(request).unwrap();
        assert_eq!(retry.token_ids, vec![22, 23, 24]);
        assert_eq!(retry.text, "7");
    }
}
