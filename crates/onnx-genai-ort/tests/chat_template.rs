use onnx_genai_ort::{ChatMessage, ChatTemplate, Tokenizer};

#[test]
fn renders_qwen_chatml_with_generation_prompt() {
    let template =
        ChatTemplate::from_model_dir(std::path::Path::new("tests/fixtures/chatml")).unwrap();
    let rendered = template
        .render(
            &[
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Hello"),
            ],
            None,
            true,
        )
        .unwrap();

    assert_eq!(
        rendered,
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn renders_tools_for_qwen_template() {
    let template =
        ChatTemplate::from_model_dir(std::path::Path::new("tests/fixtures/chatml")).unwrap();
    let rendered = template
        .render(
            &[ChatMessage::user("What time is it?")],
            Some(r#"[{"type":"function","function":{"name":"clock","description":"Get time"}}]"#),
            true,
        )
        .unwrap();

    assert!(rendered.contains("<tools>\n"));
    assert!(rendered.contains(r#""name":"clock""#));
    assert!(rendered.contains("<tool_call>"));
    assert!(rendered.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn tokenizer_surfaces_all_eos_token_ids() {
    let tokenizer = Tokenizer::from_file("tests/fixtures/chatml/tokenizer.json").unwrap();

    assert_eq!(tokenizer.eos_token_id(), Some(151645));
    assert_eq!(tokenizer.eos_token_ids(), vec![151645, 151643]);
    assert_eq!(tokenizer.token_id("<|im_start|>"), Some(151644));
    assert_eq!(tokenizer.token_id("<|im_end|>"), Some(151645));
    assert!(
        tokenizer
            .decode_with_special_tokens(&[151644, 151645])
            .unwrap()
            .contains("<|im_end|>")
    );
}
