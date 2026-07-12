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
            BenchmarkId::new("scatter_context_tokens", context_length),
            &context_length,
            |b, &context_length| {
                b.iter_batched(
                    || engine("tiny-llm-scatter"),
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

fn batch_throughput(c: &mut Criterion) {
    const NEW_TOKENS: usize = 4;
    let mut group = c.benchmark_group("model_batch");
    group.sample_size(10);
    for batch_size in [1_usize, 2, 4, 8] {
        group.throughput(Throughput::Elements((batch_size * NEW_TOKENS) as u64));
        group.bench_with_input(
            BenchmarkId::new("scatter_tokens_per_second", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter_batched(
                    || {
                        let requests = (0..batch_size)
                            .map(|index| request(vec![4 + index as u32 % 20, 5, 6], NEW_TOKENS))
                            .collect::<Vec<_>>();
                        (engine("tiny-llm-scatter"), requests)
                    },
                    |(mut engine, requests)| {
                        engine.generate_batched_static(black_box(requests)).unwrap()
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    e2e_tokens_per_second,
    prefill_latency,
    batch_throughput
);
criterion_main!(benches);
