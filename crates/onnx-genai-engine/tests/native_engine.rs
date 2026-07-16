#![cfg(feature = "native-backend")]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest,
};
use std::path::Path;

#[test]
fn engine_generates_through_explicit_native_backend() -> anyhow::Result<()> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    assert_eq!(engine.decode_backend(), EngineDecodeBackend::Native);

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
    request.options.max_new_tokens = 3;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    let mut streamed = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        streamed.push(token.token_id);
        Ok(())
    };
    let result = engine.generate_with_callback(request, Some(&mut callback))?;

    assert_eq!(result.token_ids, vec![1, 1, 1]);
    assert_eq!(streamed, result.token_ids);
    assert!(engine.create_session().is_err());
    Ok(())
}
