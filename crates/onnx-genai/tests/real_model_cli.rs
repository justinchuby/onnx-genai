use std::{path::Path, process::Command};

#[test]
#[ignore = "requires a locally built real model at models/tinystories"]
fn tinystories_cli_generates_coherent_english() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/tinystories");
    if !model_dir.join("model.onnx").is_file() || !model_dir.join("tokenizer.json").is_file() {
        eprintln!("skipping: build the real model first with scripts/build_real_model.sh");
        return Ok(());
    }

    let output = Command::new(std::env::var("CARGO_BIN_EXE_onnx-genai")?)
        .args([
            "generate",
            "--model",
            model_dir.to_str().expect("model path is valid UTF-8"),
            "--max-new-tokens",
            "30",
            "Once upon a time",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "onnx-genai generate failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8(output.stdout)?;
    assert!(
        text.contains("little girl") && text.contains("play outside"),
        "unexpected generated text: {text:?}"
    );
    assert!(
        text.split_whitespace()
            .filter(|word| word.chars().any(char::is_alphabetic))
            .count()
            >= 10,
        "generated text is too short or incoherent: {text:?}"
    );

    Ok(())
}

#[test]
#[ignore = "requires a locally built Qwen model at models/qwen2.5-0.5b"]
fn qwen_cli_generates_chatml_answer() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/qwen2.5-0.5b");
    if !model_dir.join("model.onnx").is_file() || !model_dir.join("tokenizer.json").is_file() {
        eprintln!("skipping: build the Qwen model first with scripts/build_qwen.sh");
        return Ok(());
    }

    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n";
    let output = Command::new(std::env::var("CARGO_BIN_EXE_onnx-genai")?)
        .args([
            "generate",
            "--model",
            model_dir.to_str().expect("model path is valid UTF-8"),
            "--max-new-tokens",
            "40",
            "--stop",
            "<|im_end|>",
            prompt,
        ])
        .output()?;

    assert!(
        output.status.success(),
        "onnx-genai generate failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8(output.stdout)?;
    assert!(
        text.trim().contains('4'),
        "expected Qwen to answer 2+2 coherently, got: {text:?}"
    );

    Ok(())
}
