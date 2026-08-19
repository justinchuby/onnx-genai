use onnx_genai_ort::Tokenizer;

#[test]
fn tiny_tokenizer_round_trip() {
    let tokenizer = Tokenizer::from_file("tests/fixtures/tiny-tokenizer.json").unwrap();

    let ids = tokenizer.encode("hello world").unwrap();
    assert_eq!(ids, vec![2, 3]);
    assert_eq!(tokenizer.decode(&ids).unwrap(), "hello world");
    assert_eq!(tokenizer.eos_token_id(), Some(1));
    assert_eq!(tokenizer.encode_i64("hello world").unwrap(), vec![2_i64, 3]);
    assert_eq!(tokenizer.decode_i64(&[2, 3]).unwrap(), "hello world");
}

#[test]
fn chat_template_bos_is_not_duplicated_by_the_post_processor() {
    let tokenizer = Tokenizer::from_file("tests/fixtures/tiny-tokenizer-bos.json").unwrap();

    // A chat template that already emits `bos_token` must not be given a second one.
    assert_eq!(
        tokenizer.encode("[BOS] hello world").unwrap(),
        vec![4, 2, 3]
    );
    // Prompts without a leading BOS still get the post-processor's copy.
    assert_eq!(tokenizer.encode("hello world").unwrap(), vec![4, 2, 3]);
}
