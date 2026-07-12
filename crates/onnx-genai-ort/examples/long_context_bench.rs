use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use onnx_genai_ort::{
    DecodeSession, DecodeSessionOptions, Environment, ModelDirectory, Session, SessionOptions,
    StaticCacheDecodeOptions, StaticCacheDecodeSession, Value,
};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    max_tokens: usize,
    prompt_token: i64,
    buckets: Vec<usize>,
    mode: Mode,
    rss_every: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Auto,
    Static,
    PastPresent,
}

#[derive(Debug, Default)]
struct Bucket {
    start: usize,
    end: usize,
    total: Duration,
    count: usize,
}

fn main() -> onnx_genai_ort::Result<()> {
    let args = parse_args()?;
    let env = Environment::new("long-context-bench")?;
    let model_path = resolve_model_path(&args.model)?;
    let session = Session::new(&env, &model_path, SessionOptions::default())?;

    match args.mode {
        Mode::Static => run_static(&session, &args)?,
        Mode::PastPresent => run_past_present(&session, &args)?,
        Mode::Auto => {
            if StaticCacheDecodeSession::detect(&session)?.is_some() {
                run_static(&session, &args)?;
            } else {
                run_past_present(&session, &args)?;
            }
        }
    }

    Ok(())
}

fn run_static(session: &Session, args: &Args) -> onnx_genai_ort::Result<()> {
    let signature = StaticCacheDecodeSession::detect(session)?
        .ok_or_else(|| invalid("model does not expose static-cache I/O"))?;
    let mut decode =
        StaticCacheDecodeSession::new(session, StaticCacheDecodeOptions { batch_size: 1 })?;
    let initial_buffers = decode.buffer_infos()?;
    let kv_bytes = initial_buffers
        .iter()
        .map(|info| info.numel * info.dtype.size_of())
        .sum::<usize>();
    println!(
        "mode=static-cache layers={} max_len={} kv_dim={} dtype={:?} binding={:?} kv_buffers={} kv_bytes={} initial_rss_mb={:.1}",
        signature.layers,
        signature.max_len,
        signature.kv_dim,
        signature.dtype,
        decode.binding_mode(),
        initial_buffers.len(),
        kv_bytes,
        rss_mb(),
    );
    println!("initial_buffer_ptrs={:?}", ptrs(&initial_buffers));

    decode.prefill(&[args.prompt_token], &[0])?;
    let target_steps = args.max_tokens.min(signature.max_len.saturating_sub(1));
    let mut buckets = buckets(&args.buckets, target_steps + 1);
    let mut peak_rss = rss_mb();
    let mut rss_samples = Vec::new();
    let mut token = args.prompt_token;

    for _ in 0..target_steps {
        let position = decode.current_len();
        let started = Instant::now();
        let logits = decode.step(&[token], &[position as i64])?;
        let elapsed = started.elapsed();
        record(&mut buckets, position + 1, elapsed);
        token = argmax(&logits)? as i64;
        maybe_sample_rss(
            args.rss_every,
            position + 1,
            &mut peak_rss,
            &mut rss_samples,
        );
    }

    let final_buffers = decode.buffer_infos()?;
    println!("final_len={}", decode.current_len());
    println!("final_token={token}");
    println!("final_buffer_ptrs={:?}", ptrs(&final_buffers));
    println!("buffers_stable={}", initial_buffers == final_buffers);
    println!("peak_rss_mb={peak_rss:.1}");
    println!("rss_samples={rss_samples:?}");
    print_buckets(&buckets);
    Ok(())
}

fn run_past_present(session: &Session, args: &Args) -> onnx_genai_ort::Result<()> {
    let mut decode = DecodeSession::new(session, DecodeSessionOptions::default())?;
    println!(
        "mode=past-present kv_mode={:?} initial_rss_mb={:.1}",
        decode.mode(),
        rss_mb(),
    );
    let target_steps = args.max_tokens.max(1);
    let mut buckets = buckets(&args.buckets, target_steps);
    let mut peak_rss = rss_mb();
    let mut rss_samples = Vec::new();
    let mut token = args.prompt_token;

    for position in 0..target_steps {
        let attention_mask = vec![1; position + 1];
        let started = Instant::now();
        let logits = decode.step(&[token], &attention_mask, &[position as i64])?;
        let elapsed = started.elapsed();
        record(&mut buckets, position + 1, elapsed);
        token = argmax(&logits)? as i64;
        maybe_sample_rss(
            args.rss_every,
            position + 1,
            &mut peak_rss,
            &mut rss_samples,
        );
    }

    println!("final_len={}", decode.past_len());
    println!("final_token={token}");
    println!("peak_rss_mb={peak_rss:.1}");
    println!("rss_samples={rss_samples:?}");
    print_buckets(&buckets);
    Ok(())
}

fn parse_args() -> onnx_genai_ort::Result<Args> {
    let mut model = None;
    let mut max_tokens = 2048;
    let mut prompt_token = 1;
    let mut buckets = vec![64, 256, 1024, 2048];
    let mut mode = Mode::Auto;
    let mut rss_every = 128;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
            "--max-tokens" => max_tokens = parse_usize(&mut args, "--max-tokens")?,
            "--prompt-token" => prompt_token = parse_i64(&mut args, "--prompt-token")?,
            "--buckets" => {
                let value = next_value(&mut args, "--buckets")?;
                buckets = value
                    .split(',')
                    .map(|part| {
                        part.parse::<usize>()
                            .map_err(|_| invalid(format!("invalid bucket end '{part}'")))
                    })
                    .collect::<onnx_genai_ort::Result<Vec<_>>>()?;
            }
            "--mode" => {
                mode = match next_value(&mut args, "--mode")?.as_str() {
                    "auto" => Mode::Auto,
                    "static" | "static-cache" => Mode::Static,
                    "past-present" | "past" => Mode::PastPresent,
                    other => return Err(invalid(format!("invalid mode '{other}'"))),
                };
            }
            "--rss-every" => rss_every = parse_usize(&mut args, "--rss-every")?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(invalid(format!("unknown argument '{other}'"))),
        }
    }
    buckets.sort_unstable();
    buckets.dedup();
    Ok(Args {
        model: model.ok_or_else(|| invalid("--model is required"))?,
        max_tokens,
        prompt_token,
        buckets,
        mode,
        rss_every,
    })
}

fn print_help() {
    println!(
        "Usage: cargo run -p onnx-genai-ort --example long_context_bench -- --model <DIR|ONNX> [--mode auto|static|past-present] [--max-tokens N] [--buckets 64,256,1024,2048] [--rss-every N]"
    );
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> onnx_genai_ort::Result<String> {
    args.next()
        .ok_or_else(|| invalid(format!("{flag} requires a value")))
}

fn parse_usize(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> onnx_genai_ort::Result<usize> {
    next_value(args, flag)?
        .parse()
        .map_err(|_| invalid(format!("{flag} requires a positive integer")))
}

fn parse_i64(args: &mut impl Iterator<Item = String>, flag: &str) -> onnx_genai_ort::Result<i64> {
    next_value(args, flag)?
        .parse()
        .map_err(|_| invalid(format!("{flag} requires an integer")))
}

fn resolve_model_path(path: &Path) -> onnx_genai_ort::Result<PathBuf> {
    if path.is_dir() {
        Ok(ModelDirectory::load(path)?.model_path)
    } else {
        Ok(path.to_path_buf())
    }
}

fn buckets(ends: &[usize], max_position: usize) -> Vec<Bucket> {
    let mut start = 1;
    let mut buckets = Vec::new();
    for &end in ends {
        if end >= start {
            buckets.push(Bucket {
                start,
                end: end.min(max_position),
                ..Bucket::default()
            });
            start = end + 1;
        }
    }
    if start <= max_position {
        buckets.push(Bucket {
            start,
            end: max_position,
            ..Bucket::default()
        });
    }
    buckets
}

fn record(buckets: &mut [Bucket], position: usize, elapsed: Duration) {
    if let Some(bucket) = buckets
        .iter_mut()
        .find(|bucket| position >= bucket.start && position <= bucket.end)
    {
        bucket.total += elapsed;
        bucket.count += 1;
    }
}

fn print_buckets(buckets: &[Bucket]) {
    println!("bucket_start,bucket_end,tokens,avg_ms_per_token");
    for bucket in buckets {
        if bucket.count == 0 {
            println!("{},{},0,n/a", bucket.start, bucket.end);
        } else {
            let avg_ms = bucket.total.as_secs_f64() * 1000.0 / bucket.count as f64;
            println!(
                "{},{},{},{avg_ms:.3}",
                bucket.start, bucket.end, bucket.count
            );
        }
    }
}

fn argmax(logits: &Value) -> onnx_genai_ort::Result<usize> {
    logits
        .to_vec_f32()?
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or_else(|| invalid("empty logits"))
}

fn maybe_sample_rss(
    every: usize,
    position: usize,
    peak_rss: &mut f64,
    samples: &mut Vec<(usize, f64)>,
) {
    if every != 0 && position.is_multiple_of(every) {
        let rss = rss_mb();
        *peak_rss = peak_rss.max(rss);
        samples.push((position, rss));
    }
}

fn rss_mb() -> f64 {
    let Ok(output) = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
    else {
        return f64::NAN;
    };
    let Ok(text) = String::from_utf8(output.stdout) else {
        return f64::NAN;
    };
    text.trim()
        .parse::<f64>()
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn ptrs(infos: &[onnx_genai_ort::StaticCacheBufferInfo]) -> Vec<usize> {
    infos.iter().map(|info| info.data_ptr).collect()
}

fn invalid(message: impl Into<String>) -> onnx_genai_ort::OrtError {
    onnx_genai_ort::OrtError::InvalidArgument(message.into())
}
