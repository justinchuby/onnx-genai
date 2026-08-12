//! End-to-end native-CUDA text decode for a split embedding+decoder package
//! (e.g. Muse-Glimmer-30B). Runs BOTH the embedding and the decoder graphs on
//! the pure-Rust CUDA execution provider (`onnx-runtime-ep-cuda`) — no ONNX
//! Runtime — doing greedy autoregressive decode with a growing KV cache.
//!
//! This validates real op coverage on hardware and measures steady-state decode
//! throughput. With `ONNX_GENAI_REQUIRE_CUDA=1` any unclaimed op aborts session
//! build, naming the offending nodes.
//!
//! Usage:
//!   CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_DEVICE=0 \
//!     cargo run --release -p onnx-genai-bench --features cuda,bench-native \
//!     --bin muse_decode -- --model <dir-with-genai_config.json> \
//!     --tokens 128 --warmups 2 --runs 3 --decode-skip 8 --prompt "..."

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use onnx_runtime_ir::DataType;
use onnx_runtime_session::{DevicePreference, InferenceSession, SessionBuilder, Tensor};
use tokenizers::Tokenizer;

struct Args {
    model: PathBuf,
    tokens: usize,
    warmups: usize,
    runs: usize,
    decode_skip: usize,
    device: u32,
    prompt: String,
    stop_on_eos: bool,
}

fn parse_args() -> Args {
    let mut model = PathBuf::new();
    let mut tokens = 128usize;
    let mut warmups = 2usize;
    let mut runs = 3usize;
    let mut decode_skip = 8usize;
    let mut device = 0u32;
    let mut prompt = String::from("Explain what a neural network is in two sentences.");
    let mut stop_on_eos = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(it.next().expect("--model needs a value")),
            "--tokens" => tokens = it.next().and_then(|v| v.parse().ok()).expect("--tokens N"),
            "--warmups" => warmups = it.next().and_then(|v| v.parse().ok()).expect("--warmups N"),
            "--runs" => runs = it.next().and_then(|v| v.parse().ok()).expect("--runs N"),
            "--decode-skip" => {
                decode_skip = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--decode-skip N");
            }
            "--device" => device = it.next().and_then(|v| v.parse().ok()).expect("--device N"),
            "--prompt" => prompt = it.next().expect("--prompt needs a value"),
            "--stop-on-eos" => stop_on_eos = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    if model.as_os_str().is_empty() {
        panic!("--model is required");
    }
    Args {
        model,
        tokens,
        warmups,
        runs,
        decode_skip,
        device,
        prompt,
        stop_on_eos,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Names of the decoder's KV inputs/outputs, discovered from the loaded graph so
/// no layer count is hard-coded.
struct DecoderIo {
    past_keys: Vec<String>,
    past_values: Vec<String>,
    present_keys: Vec<String>,
    present_values: Vec<String>,
    head_size: usize,
    kv_heads: usize,
}

fn discover_decoder_io(session: &InferenceSession) -> Result<DecoderIo> {
    let mut past_keys = Vec::new();
    let mut past_values = Vec::new();
    let mut head_size = 0usize;
    let mut kv_heads = 0usize;
    for io in session.inputs() {
        if io.name.ends_with(".key") && io.name.contains("past") {
            past_keys.push(io.name.clone());
        } else if io.name.ends_with(".value") && io.name.contains("past") {
            past_values.push(io.name.clone());
        }
        if (io.name.ends_with(".key") || io.name.ends_with(".value")) && kv_heads == 0 {
            let dims = &io.shape;
            if dims.len() == 4 {
                if let Some(h) = dims[1].as_static() {
                    kv_heads = h;
                }
                if let Some(d) = dims[3].as_static() {
                    head_size = d;
                }
            }
        }
    }
    let present_keys: Vec<String> = past_keys
        .iter()
        .map(|k| k.replace("past_key_values", "present"))
        .collect();
    let present_values: Vec<String> = past_values
        .iter()
        .map(|v| v.replace("past_key_values", "present"))
        .collect();
    if past_keys.is_empty() {
        bail!("decoder graph exposes no past_key_values.*.key inputs");
    }
    Ok(DecoderIo {
        past_keys,
        past_values,
        present_keys,
        present_values,
        head_size,
        kv_heads,
    })
}

fn empty_kv(kv_heads: usize, head_size: usize) -> Result<Tensor> {
    Tensor::from_raw(DataType::BFloat16, vec![1, kv_heads, 0, head_size], &[])
        .context("build empty KV tensor")
}

/// Logits row → argmax token id over the vocab. Handles f32 and bf16 logits.
fn argmax_last(logits: &Tensor, seq_len: usize, vocab: usize) -> Result<u32> {
    let start = (seq_len - 1) * vocab;
    let end = seq_len * vocab;
    let row: Vec<f32> = match logits.dtype {
        DataType::Float32 => {
            let all = logits.to_vec_f32();
            if all.len() < end {
                bail!(
                    "logits buffer too small: {} < {}*{}",
                    all.len(),
                    seq_len,
                    vocab
                );
            }
            all[start..end].to_vec()
        }
        DataType::BFloat16 => {
            let bits = logits
                .try_as_slice_u16()
                .context("could not read bf16 logits as u16")?;
            if bits.len() < end {
                bail!(
                    "logits buffer too small: {} < {}*{}",
                    bits.len(),
                    seq_len,
                    vocab
                );
            }
            // bf16 is the high 16 bits of the f32 bit pattern.
            bits[start..end]
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect()
        }
        other => bail!("unsupported logits dtype {other:?}"),
    };
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    Ok(best as u32)
}

fn output_index(session: &InferenceSession, name: &str) -> Result<usize> {
    session
        .outputs()
        .iter()
        .position(|o| o.name == name)
        .with_context(|| format!("decoder output '{name}' not found"))
}

struct RunResult {
    token_ids: Vec<u32>,
    text: String,
    token_times: Vec<Duration>,
}

fn embed(
    embedding: &mut InferenceSession,
    embed_in: &str,
    image_in: Option<&str>,
    ids: &[i64],
) -> Result<Tensor> {
    let n = ids.len();
    let input_ids = Tensor::from_i64(&[1, n], ids)?;
    let mut inputs: Vec<(&str, &Tensor)> = vec![(embed_in, &input_ids)];
    let image = if image_in.is_some() {
        Some(Tensor::from_raw(DataType::BFloat16, vec![0, 6656], &[])?)
    } else {
        None
    };
    if let (Some(name), Some(t)) = (image_in, image.as_ref()) {
        inputs.push((name, t));
    }
    let mut out = embedding.run(&inputs).context("embedding run")?;
    Ok(out.remove(0))
}

#[allow(clippy::too_many_arguments)]
fn generate(
    embedding: &mut InferenceSession,
    decoder: &mut InferenceSession,
    tokenizer: &Tokenizer,
    io: &DecoderIo,
    embed_in: &str,
    image_in: Option<&str>,
    logits_idx: usize,
    present_key_idx: &[usize],
    present_value_idx: &[usize],
    vocab: usize,
    prompt_ids: &[i64],
    max_new: usize,
    eos: Option<u32>,
) -> Result<RunResult> {
    let start = Instant::now();
    let mut token_times = Vec::with_capacity(max_new);
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);

    // Prefill.
    let mut inputs_embeds = embed(embedding, embed_in, image_in, prompt_ids)?;
    let mut total_len = prompt_ids.len();
    let mut attention_mask = Tensor::from_i64(&[1, total_len], &vec![1i64; total_len])?;
    let mut past_k: Vec<Tensor> = (0..io.past_keys.len())
        .map(|_| empty_kv(io.kv_heads, io.head_size))
        .collect::<Result<_>>()?;
    let mut past_v: Vec<Tensor> = (0..io.past_values.len())
        .map(|_| empty_kv(io.kv_heads, io.head_size))
        .collect::<Result<_>>()?;

    for step in 0..max_new {
        let seq_len = if step == 0 { prompt_ids.len() } else { 1 };
        let mut binds: Vec<(&str, &Tensor)> = Vec::with_capacity(2 + past_k.len() * 2);
        binds.push(("inputs_embeds", &inputs_embeds));
        binds.push(("attention_mask", &attention_mask));
        for (i, name) in io.past_keys.iter().enumerate() {
            binds.push((name, &past_k[i]));
        }
        for (i, name) in io.past_values.iter().enumerate() {
            binds.push((name, &past_v[i]));
        }
        let outputs = decoder.run(&binds).context("decoder run")?;
        token_times.push(start.elapsed());

        let next = argmax_last(&outputs[logits_idx], seq_len, vocab)?;
        generated.push(next);

        // Rotate present -> past for the next step.
        let mut new_k = Vec::with_capacity(past_k.len());
        let mut new_v = Vec::with_capacity(past_v.len());
        for i in 0..past_k.len() {
            new_k.push(outputs[present_key_idx[i]].clone());
            new_v.push(outputs[present_value_idx[i]].clone());
        }
        past_k = new_k;
        past_v = new_v;

        if eos == Some(next) {
            break;
        }
        // Prepare next-step embedding + mask.
        if step + 1 < max_new {
            inputs_embeds = embed(embedding, embed_in, image_in, &[next as i64])?;
            total_len += 1;
            attention_mask = Tensor::from_i64(&[1, total_len], &vec![1i64; total_len])?;
        }
    }

    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
    Ok(RunResult {
        token_ids: generated,
        text,
        token_times,
    })
}

fn main() -> Result<()> {
    let args = parse_args();
    let model_dir = &args.model;

    // genai_config for vocab / eos / bos.
    let genai: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("genai_config.json"))?)
            .context("parse genai_config.json")?;
    let vocab = genai["model"]["vocab_size"].as_u64().unwrap_or(202048) as usize;
    let eos = genai["model"]["eos_token_id"].as_u64().map(|v| v as u32);
    let bos = genai["model"]["bos_token_id"].as_u64().map(|v| v as i64);

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let device = DevicePreference::Gpu {
        index: Some(args.device),
    };
    println!(
        "muse_decode: loading embedding + decoder on CUDA device {}",
        args.device
    );
    let load_start = Instant::now();
    let mut embedding = SessionBuilder::new()
        .model(model_dir.join("embedding/model.onnx"))
        .device(device.clone())
        .build()
        .context("load embedding on CUDA")?;
    let mut decoder = SessionBuilder::new()
        .model(model_dir.join("decoder/model.onnx"))
        .device(device.clone())
        .build()
        .context("load decoder on CUDA")?;
    println!(
        "muse_decode: loaded in {:.1} s",
        load_start.elapsed().as_secs_f64()
    );

    // Discover I/O.
    let embed_in = embedding
        .inputs()
        .iter()
        .find(|i| i.name.contains("input_ids"))
        .map(|i| i.name.clone())
        .unwrap_or_else(|| "input_ids".to_string());
    let image_in = embedding
        .inputs()
        .iter()
        .find(|i| i.name.contains("image"))
        .map(|i| i.name.clone());
    let io = discover_decoder_io(&decoder)?;
    let logits_idx = output_index(&decoder, "logits")?;
    let present_key_idx: Vec<usize> = io
        .present_keys
        .iter()
        .map(|n| output_index(&decoder, n))
        .collect::<Result<_>>()?;
    let present_value_idx: Vec<usize> = io
        .present_values
        .iter()
        .map(|n| output_index(&decoder, n))
        .collect::<Result<_>>()?;
    println!(
        "muse_decode: layers={} kv_heads={} head_size={} vocab={}",
        io.past_keys.len(),
        io.kv_heads,
        io.head_size,
        vocab
    );

    // Prompt token ids (raw + optional BOS). Kept simple/deterministic for a
    // coherence smoke test; greedy decode is deterministic.
    let enc = tokenizer
        .encode(args.prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
    let mut prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
    if let Some(b) = bos {
        prompt_ids.insert(0, b);
    }
    println!("muse_decode: prompt_tokens={}", prompt_ids.len());

    let eos_stop = if args.stop_on_eos { eos } else { None };

    // Warmups.
    for w in 0..args.warmups {
        let r = generate(
            &mut embedding,
            &mut decoder,
            &tokenizer,
            &io,
            &embed_in,
            image_in.as_deref(),
            logits_idx,
            &present_key_idx,
            &present_value_idx,
            vocab,
            &prompt_ids,
            args.tokens,
            eos_stop,
        )?;
        if w == 0 {
            println!("generated_text (warmup): {:?}", r.text);
            println!(
                "first_token_ids: {:?}",
                &r.token_ids[..r.token_ids.len().min(16)]
            );
        }
    }

    // Measured runs.
    let mut prefills_ms = Vec::with_capacity(args.runs);
    let mut decode_ms_per_token = Vec::with_capacity(args.runs);
    let mut throughputs = Vec::with_capacity(args.runs);
    let mut reference: Option<Vec<u32>> = None;
    for run in 1..=args.runs {
        let r = generate(
            &mut embedding,
            &mut decoder,
            &tokenizer,
            &io,
            &embed_in,
            image_in.as_deref(),
            logits_idx,
            &present_key_idx,
            &present_value_idx,
            vocab,
            &prompt_ids,
            args.tokens,
            eos_stop,
        )?;
        if r.token_times.len() <= args.decode_skip {
            bail!("not enough tokens for --decode-skip");
        }
        if let Some(ref_ids) = &reference {
            if ref_ids != &r.token_ids {
                eprintln!("warning: greedy decode not deterministic across runs");
            }
        } else {
            reference = Some(r.token_ids.clone());
            println!("generated_text: {:?}", r.text);
        }
        let prefill_ms = r.token_times[0].as_secs_f64() * 1000.0;
        let decode_tokens = r.token_times.len() - args.decode_skip;
        let baseline = if args.decode_skip == 0 {
            Duration::ZERO
        } else {
            r.token_times[args.decode_skip - 1]
        };
        let decode_wall = r.token_times[r.token_times.len() - 1] - baseline;
        let ms_per = decode_wall.as_secs_f64() * 1000.0 / decode_tokens as f64;
        let tok_s = decode_tokens as f64 / decode_wall.as_secs_f64();
        println!(
            "run {run}: prefill={prefill_ms:.1} ms decode_tokens={decode_tokens} \
             decode={ms_per:.2} ms/token throughput={tok_s:.2} tok/s"
        );
        prefills_ms.push(prefill_ms);
        decode_ms_per_token.push(ms_per);
        throughputs.push(tok_s);
    }
    let mut tp = throughputs.clone();
    let lo = tp.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = tp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "median: prefill={:.1} ms decode={:.2} ms/token throughput={:.2} tok/s (min={:.2} max={:.2}, runs={} warmups={} decode_skip={})",
        median(&mut prefills_ms),
        median(&mut decode_ms_per_token),
        median(&mut tp),
        lo,
        hi,
        args.runs,
        args.warmups,
        args.decode_skip
    );
    Ok(())
}
