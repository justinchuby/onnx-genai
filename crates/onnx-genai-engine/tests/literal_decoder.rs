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
        // Cold Engine callbacks publish only after transaction commit. Cancel
        // from the real per-forward hook: two earlier forwards selected C3/A9;
        // the third advances KV but must abort before decoding FF/committing.
        let forwards = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = forwards.clone();
        let control = onnx_genai_engine::GenerationControl::new();
        let cancel = control.clone();
        let control = control.with_checkpoint(move |boundary| {
            if boundary == onnx_genai_engine::GenerationBoundary::AfterDecoderForward {
                let count = observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                if count == 3 {
                    assert!(cancel.cancel());
                }
            }
            Ok(())
        });
        let mut published = Vec::new();
        let mut callback = |token: onnx_genai_engine::GenerateToken| {
            published.push((token.token_id, token.text));
            Ok(())
        };
        let error = engine
            .generate_with_control_callbacks(request.clone(), control, None, Some(&mut callback))
            .expect_err(
                "cancel must abort while the two-token byte/replacement prefix is provisional",
            );
        assert!(
            error.is::<onnx_genai_engine::GenerationCancelled>(),
            "byte_level={byte_level}: {error:#}"
        );
        assert_eq!(
            forwards.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "byte_level={byte_level}: cancel must follow two selected tokens and third real forward"
        );
        assert!(
            published.is_empty(),
            "byte_level={byte_level}: aborted cold turn published {published:?}"
        );
        let retry = engine.generate(request).unwrap();
        assert_eq!(retry.token_ids, vec![22, 23, 24]);
        assert_eq!(retry.text, "7");
    }
}

fn native_decoded_text_constraint_roundtrip(lark: bool) {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for split in [false, true] {
        let root = tempfile::tempdir().unwrap();
        decoder::build(
            root.path(),
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
            split,
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
        let constraint = |impossible: bool| {
            if lark {
                GenerateConstraint::Lark {
                    grammar: format!("start: {:?}", if impossible { "~" } else { "7" }),
                    prefix: String::new(),
                }
            } else {
                GenerateConstraint::Regex {
                    pattern: if impossible { "~".into() } else { "7".into() },
                    prefix: String::new(),
                }
            }
        };
        let mut good = request(4096);
        good.options.constraint = Some(constraint(false));
        good.options.max_new_tokens = if split { 2 } else { 1 };
        good.options.logit_bias.insert(
            if split { 24 } else { 23 },
            onnx_genai_engine::logits::LogitBias::Add(1002.),
        );
        for failed in [false, true, false] {
            let mut current = good.clone();
            current.options.constraint = Some(constraint(failed));
            let forwards = Arc::new(AtomicUsize::new(0));
            let observed = forwards.clone();
            let control =
                onnx_genai_engine::GenerationControl::new().with_checkpoint(move |boundary| {
                    if boundary == onnx_genai_engine::GenerationBoundary::AfterDecoderForward {
                        observed.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(())
                });
            let mut published = Vec::new();
            let mut callback = |token: onnx_genai_engine::GenerateToken| {
                published.push(token.text);
                Ok(())
            };
            let result =
                engine.generate_with_control_callbacks(current, control, None, Some(&mut callback));
            assert!(
                forwards.load(Ordering::SeqCst) > 0,
                "lark={lark}, split={split}, failed={failed}: no actual Native decoder execution"
            );
            if failed {
                let error = result
                    .expect_err("unreachable decoded constraint must reject the actual vocabulary");
                assert!(
                    format!("{error:#}").contains("no finite selectable token"),
                    "lark={lark}, split={split}: {error:#}"
                );
                assert!(
                    published.is_empty(),
                    "aborted cold turn published {published:?}"
                );
            } else {
                let result = result.unwrap();
                assert_eq!(result.text, "7");
                assert_eq!(
                    result.token_ids,
                    if split { vec![22, 23] } else { vec![22] }
                );
                assert_eq!(published.concat(), "7");
            }
        }
    }
}
#[test]
fn native_literal_decoded_regex_masks_real_tokens_rejects_unreachable_output_and_retries() {
    native_decoded_text_constraint_roundtrip(false);
}
#[test]
fn native_literal_decoded_lark_masks_real_tokens_rejects_unreachable_output_and_retries() {
    native_decoded_text_constraint_roundtrip(true);
}
