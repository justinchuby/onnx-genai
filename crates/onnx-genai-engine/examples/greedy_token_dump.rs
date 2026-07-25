//! Greedy token-id dump for parity audits.
//!
//! Usage: greedy_token_dump <model_dir> <prompt> <max_new_tokens>
//! Prints the greedy-decoded generated token ids as a JSON array so a
//! BEFORE/AFTER diff is exact and load-independent.

use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().expect("model_dir");
    let prompt = args.next().expect("prompt");
    let max_new_tokens: usize = args.next().expect("max_new_tokens").parse()?;

    let mut options = GenerateOptions {
        max_new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    options.top_p = 1.0;
    options.top_k = 0;

    let request = GenerateRequest {
        prompt: GeneratePrompt::Text(prompt),
        options,
    };

    let mut engine = Engine::from_dir(std::path::Path::new(&model_dir), EngineConfig::default())?;
    let result = engine.generate(request)?;
    let ids: Vec<String> = result.token_ids.iter().map(|id| id.to_string()).collect();
    println!("[{}]", ids.join(","));
    Ok(())
}
