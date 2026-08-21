//! Temporary CPU-vs-CUDA divergence probe for real DeepSeek-V2-Lite int4.
//! Runs the same greedy prompt on native CPU and native CUDA and reports the
//! first differing generated-token index. Investigation harness only.
#![cfg(feature = "native-cuda")]

use std::path::PathBuf;

use onnx_genai_engine::{
    DecodePrecision, Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice,
};

const PROMPT: &str = "Hello";
const TOKENS: usize = 8;

fn generate(model_dir: &std::path::Path, device: NativeDecodeDevice) -> anyhow::Result<Vec<u32>> {
    let mut engine = Engine::from_dir(
        model_dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(device),
            decode_precision: DecodePrecision::Model,
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::Text(PROMPT.to_string()));
    request.options.max_new_tokens = TOKENS;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    Ok(engine.generate(request)?.token_ids)
}

#[test]
#[ignore = "requires the real DeepSeek-V2-Lite int4 export and a CUDA device"]
fn deepseek_v2_lite_cpu_vs_cuda() -> anyhow::Result<()> {
    let Some(model_dir) = std::env::var_os("DEEPSEEK_V2_LITE_CUDA_DIR").map(PathBuf::from) else {
        eprintln!("skipping: set DEEPSEEK_V2_LITE_CUDA_DIR");
        return Ok(());
    };
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }
    let cpu = generate(&model_dir, NativeDecodeDevice::Cpu)?;
    let cuda = generate(&model_dir, NativeDecodeDevice::Cuda { index: Some(0) })?;
    eprintln!("CPU : {cpu:?}");
    eprintln!("CUDA: {cuda:?}");
    let first = cpu.iter().zip(&cuda).position(|(a, b)| a != b);
    match first {
        Some(idx) => eprintln!("FIRST DIVERGENCE at generated index {idx}"),
        None => eprintln!("MATCH — no divergence in first {TOKENS} tokens"),
    }
    Ok(())
}
