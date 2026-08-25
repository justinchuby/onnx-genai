use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use onnx_genai_bench::fixture_path;
use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::SessionOptions;

fn request(prompt: impl Into<GeneratePrompt>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt);
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

fn engine(fixture: &str) -> Engine {
    let fixture = fixture_path(fixture);
    Engine::from_dir_with_session_options(
        &fixture,
        EngineConfig::default(),
        SessionOptions::default().with_intra_op_threads(1),
    )
    .expect("tiny model fixture must load")
}

fn e2e_tokens_per_second(c: &mut Criterion) {
    const NEW_TOKENS: usize = 8;
    let mut group = c.benchmark_group("model_e2e");
    group.sample_size(10);
    group.throughput(Throughput::Elements(NEW_TOKENS as u64));
    group.bench_function("tiny_llm_tokens_per_second", |b| {
        b.iter_batched(
            || engine("tiny-llm"),
            |mut engine| {
                engine
                    .generate(black_box(request("hello world", NEW_TOKENS)))
                    .unwrap()
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

fn prefill_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_prefill");
    group.sample_size(10);
    for context_length in [1_usize, 4, 8, 12] {
        group.bench_with_input(
            BenchmarkId::new("context_tokens", context_length),
            &context_length,
            |b, &context_length| {
                b.iter_batched(
                    || engine("tiny-llm"),
                    |mut engine| {
                        let prompt = (0..context_length)
                            .map(|index| 4 + (index as u32 % 20))
                            .collect::<Vec<_>>();
                        engine.generate(black_box(request(prompt, 1))).unwrap()
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}

fn prefix_cache_prefill(c: &mut Criterion) {
    // A shared 12-token prefix followed by a short unique suffix; prefill of the
    // shared prefix is what the KV prefix cache can skip on a warm engine.
    let shared_prefix = (0..12)
        .map(|index| 4 + (index as u32 % 20))
        .collect::<Vec<_>>();
    let full_prompt = {
        let mut prompt = shared_prefix.clone();
        prompt.extend_from_slice(&[24, 25, 26]);
        prompt
    };

    let mut group = c.benchmark_group("model_prefix_cache");
    group.sample_size(10);

    // Cold: a fresh engine with an empty prefix cache pays full prefill.
    group.bench_function("cold_prefill", |b| {
        b.iter_batched(
            || (engine("tiny-llm"), full_prompt.clone()),
            |(mut engine, prompt)| engine.generate(black_box(request(prompt, 1))).unwrap(),
            BatchSize::LargeInput,
        )
    });

    // Warm: the engine has already decoded the shared prefix, so a follow-up
    // prompt sharing that prefix can reuse the cached KV and skip most prefill.
    group.bench_function("warm_prefill", |b| {
        b.iter_batched(
            || {
                let mut engine = engine("tiny-llm");
                engine
                    .generate(request(shared_prefix.clone(), 1))
                    .expect("warm-up generate must succeed");
                (engine, full_prompt.clone())
            },
            |(mut engine, prompt)| engine.generate(black_box(request(prompt, 1))).unwrap(),
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    e2e_tokens_per_second,
    prefill_latency,
    prefix_cache_prefill
);
criterion_main!(benches);
