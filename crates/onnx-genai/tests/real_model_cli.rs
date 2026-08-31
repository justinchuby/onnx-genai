use std::{path::Path, process::Command};

fn cli(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO"));
    command
        .current_dir(root)
        .args(["run", "--quiet", "-p", "onnx-genai-cli", "--"]);
    command
}

#[test]
#[ignore = "requires a locally built real model at models/tinystories"]
fn tinystories_cli_generates_coherent_english() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/tinystories");
    if !model_dir.join("model.onnx").is_file() || !model_dir.join("tokenizer.json").is_file() {
        eprintln!("skipping: build the real model first with scripts/build_real_model.sh");
        return Ok(());
    }

    let output = cli(&root)
        .args([
            "generate",
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
#[ignore = "real-model test: run scripts/build_qwen.sh to create models/qwen2.5-0.5b"]
fn qwen_cli_applies_chat_template_and_generates_deterministically_on_cpu_ort() -> anyhow::Result<()>
{
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/qwen2.5-0.5b");
    for required_file in ["model.onnx", "tokenizer.json", "tokenizer_config.json"] {
        let path = model_dir.join(required_file);
        let display = path.display();
        assert!(
            path.is_file(),
            "missing real-model prerequisite {display}: run scripts/build_qwen.sh first"
        );
    }

    let output = cli(&root)
        .env("ONNX_GENAI_EP", "cpu")
        .args([
            "generate",
            model_dir.to_str().expect("model path is valid UTF-8"),
            "--backend",
            "ort",
            "--max-new-tokens",
            "8",
            "--temperature",
            "0",
            "--stop",
            "<|im_end|>",
            "Choose any decimal number from 0 through 9. Reply with only the number.",
        ])
        .output()?;

    let stdout = String::from_utf8(output.stdout)?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "onnx-genai generate failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("could not load chat template"),
        "Qwen's chat template was not applied; stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("missing from input feed"),
        "ORT reported an undeclared required graph input; stderr:\n{stderr}"
    );

    let answer = stdout.trim();
    assert!(!answer.is_empty(), "Qwen returned an empty answer");
    let number = answer.parse::<f64>().unwrap_or_else(|error| {
        panic!("Qwen did not follow the single-number response contract: {answer:?}: {error}")
    });
    assert!(
        number.is_finite() && (0.0..=9.0).contains(&number),
        "Qwen returned a number outside the requested range: {answer:?}"
    );

    Ok(())
}
