use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use onnx_genai_bench::{
    grammar_constraint, logit_processor_chain, processor_context, synthetic_logits, tokenizer,
};
use onnx_genai_engine::logits::{
    Constraint, LogitProcessor, MinPProcessor, TokenCandidate, TopKProcessor, TopPProcessor,
};
use onnx_genai_engine::{CategoricalSampler, GreedySampler, Sampler};
use onnx_genai_kv::{KvCacheOps, PagedKvCache};

fn tokenization(c: &mut Criterion) {
    let tokenizer = tokenizer();
    let text = "the quick brown fox jumps over the lazy dog. ".repeat(128);
    let encoded = tokenizer.encode(text.clone(), false).unwrap();
    let token_ids = encoded.get_ids().repeat(128);
    let mut group = c.benchmark_group("tokenization");

    group.throughput(Throughput::Elements(encoded.len() as u64));
    group.bench_function("encode_tokens_per_second", |b| {
        b.iter(|| tokenizer.encode(black_box(text.as_str()), false).unwrap())
    });

    group.throughput(Throughput::Elements(token_ids.len() as u64));
    group.bench_function("decode_tokens_per_second", |b| {
        b.iter(|| tokenizer.decode(black_box(&token_ids), false).unwrap())
    });
    group.finish();
}

fn sampling(c: &mut Criterion) {
    let logits = synthetic_logits();
    let context = processor_context();
    let mut group = c.benchmark_group("sampling_latency");
    group.throughput(Throughput::Elements(1));

    group.bench_function("greedy_per_token", |b| {
        let mut sampler = GreedySampler;
        b.iter(|| sampler.sample(black_box(&logits), black_box(&context)))
    });

    let policies: [(&str, Box<dyn LogitProcessor>); 3] = [
        ("top_k_per_token", Box::new(TopKProcessor { top_k: 50 })),
        ("top_p_per_token", Box::new(TopPProcessor { top_p: 0.9 })),
        ("min_p_per_token", Box::new(MinPProcessor { min_p: 0.05 })),
    ];
    for (name, processor) in policies {
        group.bench_function(name, |b| {
            let mut sampler = CategoricalSampler::new(0.42);
            b.iter_batched(
                || logits.clone(),
                |mut work| {
                    processor.process(&mut work, &context);
                    sampler.sample(black_box(&work), black_box(&context))
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn kv_cache(c: &mut Criterion) {
    const TOKENS: usize = 256;
    const PAGE_SIZE: usize = 16;
    let mut cache = PagedKvCache::new(PAGE_SIZE, 64);
    let mut group = c.benchmark_group("kv_cache");
    group.throughput(Throughput::Elements((TOKENS / PAGE_SIZE) as u64));
    group.bench_function("alloc_dealloc_pages", |b| {
        b.iter(|| {
            let sequence = cache.create_sequence();
            cache.append(sequence, TOKENS).unwrap();
            cache.remove(sequence).unwrap();
        })
    });
    group.finish();
}

fn logit_processing(c: &mut Criterion) {
    let logits = synthetic_logits();
    let context = processor_context();
    let chain = logit_processor_chain();
    let mut group = c.benchmark_group("logit_processing");
    group.throughput(Throughput::Elements(1));
    group.bench_function("seven_processor_chain_per_step", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut work| chain.process(black_box(&mut work), black_box(&context)),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn grammar_masking(c: &mut Criterion) {
    let tokenizer = tokenizer();
    let constraint = grammar_constraint(&tokenizer);
    let vocab = tokenizer.get_vocab(true);
    let mut token_texts = vec![String::new(); tokenizer.get_vocab_size(false)];
    for (text, id) in vocab {
        if let Some(slot) = token_texts.get_mut(id as usize) {
            *slot = text;
        }
    }
    let candidates = token_texts
        .into_iter()
        .enumerate()
        .map(|(token_id, text)| TokenCandidate {
            token_id: token_id as u32,
            text,
            is_eos: token_id == 3,
        })
        .collect::<Vec<_>>();
    let context = processor_context();
    let mut group = c.benchmark_group("grammar_masking");
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(
        BenchmarkId::new("llguidance_compute_mask", candidates.len()),
        &candidates,
        |b, candidates| {
            b.iter(|| constraint.allowed_next_tokens(black_box(&context), black_box(candidates)))
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    tokenization,
    sampling,
    kv_cache,
    logit_processing,
    grammar_masking
);
criterion_main!(benches);
