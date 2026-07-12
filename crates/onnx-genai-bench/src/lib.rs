use std::path::PathBuf;

use onnx_genai_engine::logits::{
    FrequencyPenaltyProcessor, LlguidanceConstraint, MinPProcessor, PresencePenaltyProcessor,
    ProcessorChain, RepetitionPenaltyProcessor, TemperatureProcessor, TopKProcessor, TopPProcessor,
};
use onnx_genai_engine::{ProcessorContext, TokenId};
use tokenizers::Tokenizer;

pub const VOCAB_SIZE: usize = 32_000;

pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

pub fn tokenizer() -> Tokenizer {
    Tokenizer::from_file(fixture_path("tiny-llm").join("tokenizer.json"))
        .expect("committed tiny tokenizer must load")
}

pub fn synthetic_logits() -> Vec<f32> {
    (0..VOCAB_SIZE)
        .map(|index| ((index * 17 % 997) as f32 - 498.0) / 73.0)
        .collect()
}

pub fn processor_context() -> ProcessorContext {
    ProcessorContext {
        prompt_tokens: (0..128).map(|token| token as TokenId % 97).collect(),
        generated_tokens: (0..64).map(|token| token as TokenId % 31).collect(),
        generated_text: "hello world ".repeat(8),
        step: 64,
    }
}

pub fn logit_processor_chain() -> ProcessorChain {
    let mut chain = ProcessorChain::new();
    chain.add(Box::new(RepetitionPenaltyProcessor { penalty: 1.1 }));
    chain.add(Box::new(FrequencyPenaltyProcessor {
        frequency_penalty: 0.2,
    }));
    chain.add(Box::new(PresencePenaltyProcessor {
        presence_penalty: 0.1,
    }));
    chain.add(Box::new(TemperatureProcessor { temperature: 0.8 }));
    chain.add(Box::new(TopKProcessor { top_k: 50 }));
    chain.add(Box::new(TopPProcessor { top_p: 0.9 }));
    chain.add(Box::new(MinPProcessor { min_p: 0.05 }));
    chain
}

pub fn grammar_constraint(tokenizer: &Tokenizer) -> LlguidanceConstraint {
    let vocab = tokenizer.get_vocab(true);
    let mut token_texts = vec![None; tokenizer.get_vocab_size(false)];
    for (text, id) in vocab {
        if let Some(slot) = token_texts.get_mut(id as usize) {
            *slot = Some(text);
        }
    }
    LlguidanceConstraint::from_token_texts(
        onnx_genai_engine::logits::GrammarConstraintKind::Regex,
        "hello( world)*",
        &token_texts,
        Some(3),
    )
    .expect("tiny tokenizer must support the benchmark grammar")
}
