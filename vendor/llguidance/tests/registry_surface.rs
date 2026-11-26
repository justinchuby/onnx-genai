//! Public APIs required by registry-only nxrt engine consumers.
use onnx_genai_llguidance::{
    api::{ParserLimits, TopLevelGrammar},
    toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie},
    GrammarBuilder, JsonCompileOptions, Matcher, ParserFactory, UnsatisfiableSchemaError,
};
use serde_json::{json, Value};
use std::sync::Arc;

#[test]
fn public_unsatisfiability_remains_distinct_from_invalid_schema_errors() {
    for schema in [
        json!(false),
        json!({"type":"object","properties":{"x":false},"required":["x"]}),
    ] {
        let error = JsonCompileOptions::default()
            .json_to_llg_with_overrides(GrammarBuilder::new(None, ParserLimits::default()), schema)
            .err()
            .expect("backend must prove the impossible schema empty");
        assert!(error.downcast_ref::<UnsatisfiableSchemaError>().is_some());
    }
    let error = JsonCompileOptions::default()
        .json_to_llg_with_overrides(
            GrammarBuilder::new(None, ParserLimits::default()),
            json!({"$ref":"#/$defs/missing"}),
        )
        .err()
        .expect("unresolved references are errors, not impossible branches");
    assert!(error.downcast_ref::<UnsatisfiableSchemaError>().is_none());
}

fn accepts(schema: Value, text: &str) -> bool {
    let mut pieces = (0..u8::MAX).map(|byte| vec![byte]).collect::<Vec<_>>();
    pieces.push(Vec::new());
    let eos = u32::try_from(pieces.len()).unwrap();
    let mut eos_piece = vec![TokTrie::SPECIAL_TOKEN_MARKER];
    eos_piece.extend_from_slice(b"<eos>");
    pieces.push(eos_piece);
    let info = TokRxInfo {
        vocab_size: u32::try_from(pieces.len()).unwrap(),
        tok_eos: eos,
        tok_bos: None,
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &pieces)));
    let factory = ParserFactory::new(&env, InferenceCapabilities::default(), &[]).unwrap();
    let mut matcher =
        Matcher::new(factory.create_parser(TopLevelGrammar::from_json_schema(schema)));
    let tokens = text.bytes().map(u32::from).collect::<Vec<_>>();
    matcher.try_consume_tokens(&tokens).unwrap() == tokens.len() && matcher.is_accepting().unwrap()
}

#[test]
fn registry_compiler_preserves_known_key_lexeme_ownership() {
    // JSON permits raw DEL. It must retain its known-key schema instead of
    // slipping through the additional-properties lexeme with a different value.
    // The engine separately normalizes arbitrary Unicode escape spellings.
    let schema = json!({"type":"object","properties":{"\u{7f}":{"const":1}},"additionalProperties":{"const":2}});
    let known = |value| format!("{{\"{}\":{value}}}", '\u{7f}');
    assert!(accepts(schema.clone(), &known(1)));
    assert!(!accepts(schema.clone(), &known(2)));
    assert!(accepts(schema.clone(), r#"{"other":2}"#));
    assert!(!accepts(schema, r#"{"other":1}"#));
}
