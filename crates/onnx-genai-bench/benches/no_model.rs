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

fn full_sort_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut sorted: Vec<f32> = logits
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    if sorted.is_empty() {
        return;
    }
    sorted.sort_unstable_by(|left, right| {
        right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = sorted[top_k.saturating_sub(1).min(sorted.len() - 1)];
    for logit in logits {
        if logit.is_nan() || *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn full_sort_top_p(logits: &mut [f32], top_p: f32) {
    if !top_p.is_finite() || top_p >= 1.0 || logits.is_empty() {
        return;
    }
    let maximum_logit = logits
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !maximum_logit.is_finite() {
        return;
    }
    let weights: Vec<_> = logits
        .iter()
        .map(|&logit| {
            if logit.is_nan() {
                0.0
            } else {
                (logit - maximum_logit).exp()
            }
        })
        .collect();
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return;
    }
    let probabilities = weights
        .into_iter()
        .enumerate()
        .map(|(index, weight)| (index, weight / total));
    let mut ranked: Vec<_> = probabilities.collect();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let cutoff = top_p.max(0.0);
    let mut cumulative = 0.0_f32;
    let mut keep_count = 0;
    for &(_, probability) in &ranked {
        keep_count += 1;
        cumulative += probability;
        if cumulative >= cutoff {
            break;
        }
    }
    for &(index, _) in ranked.iter().skip(keep_count) {
        logits[index] = f32::NEG_INFINITY;
    }
}

fn qwen3_sampling_logits() -> Vec<f32> {
    const VOCAB_SIZE: usize = 151_936;
    (0..VOCAB_SIZE)
        .map(|index| {
            let mixed = index.wrapping_mul(1_103_515_245).wrapping_add(12_345) % 1_048_573;
            mixed as f32 / 65_536.0 - 8.0
        })
        .collect()
}

fn qwen3_sampling_processors(c: &mut Criterion) {
    let logits = qwen3_sampling_logits();
    let context = processor_context();
    let top_k = TopKProcessor { top_k: 20 };
    let top_p = TopPProcessor { top_p: 0.95 };

    let mut top_k_logits = logits.clone();
    top_k.process(&mut top_k_logits, &context);

    let mut group = c.benchmark_group("qwen3_sampling_processors");
    group.throughput(Throughput::Elements(1));
    group.bench_function("top_k_full_sort_baseline", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut work| full_sort_top_k(black_box(&mut work), 20),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("top_k_partial_selection", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut work| top_k.process(black_box(&mut work), black_box(&context)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("top_p_full_sort_after_top_k_baseline", |b| {
        b.iter_batched(
            || top_k_logits.clone(),
            |mut work| full_sort_top_p(black_box(&mut work), 0.95),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("top_p_fast_after_top_k", |b| {
        b.iter_batched(
            || top_k_logits.clone(),
            |mut work| top_p.process(black_box(&mut work), black_box(&context)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("top_k_top_p_full_sort_baseline", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut work| {
                full_sort_top_k(black_box(&mut work), 20);
                full_sort_top_p(black_box(&mut work), 0.95);
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("top_k_top_p_fast", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut work| {
                top_k.process(black_box(&mut work), black_box(&context));
                top_p.process(black_box(&mut work), black_box(&context));
            },
            BatchSize::SmallInput,
        )
    });
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
    qwen3_sampling_processors,
    kv_cache,
    logit_processing,
    grammar_masking
);
criterion_main!(benches);
