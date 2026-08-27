#![allow(dead_code)]

use anyhow::ensure;
use onnx_genai_engine::{
    AdapterSelection, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};

fn options(max_new_tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::default();
    options.max_new_tokens = max_new_tokens;
    options.seed = Some(7);
    options
}

pub fn tiny_png() -> Vec<u8> {
    vec![
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2,
        0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 0, 0,
        3, 1, 1, 0, 201, 254, 146, 239, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ]
}

pub fn speech_request(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenRows(vec![prompt_tokens.to_vec()]),
        options: options(max_new_tokens),
    }))
}

pub fn vlm_request(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let png = tiny_png();
    let png_len = i64::try_from(png.len())?;
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(prompt_tokens.to_vec()),
        options: options(max_new_tokens),
    })
    .with_input(
        "request.image",
        Value::from_raw_bytes(png, &[png_len], DataType::Uint8)?,
    ))
}

pub fn adapter_request(
    active: &[bool],
    values: &[f32],
    selection: AdapterSelection,
) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = selection.rows.len();
    ensure!(
        values.len() == rows * 2,
        "adapter activations must contain exactly 2 features per row"
    );
    ensure!(
        active.len() == rows,
        "adapter active mask must contain one flag per row"
    );
    let batch = i64::try_from(rows)?;
    let mut segments = vec![-1i64; rows * 2];
    let mut adapter_counts = vec![0i64; rows];
    let mut adapter_scales = vec![0.0f32; rows * 2];
    for (row, activations) in selection.rows.iter().enumerate() {
        adapter_counts[row] = i64::try_from(activations.len())?;
        for (slot, activation) in activations.iter().enumerate() {
            segments[row * 2 + slot] = match activation.adapter.as_str() {
                "blue" => 0,
                "green" => 1,
                "peft" => 2,
                "red" => 3,
                other => anyhow::bail!("unknown test adapter {other}"),
            };
            adapter_scales[row * 2 + slot] = activation.scale;
        }
    }
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: Default::default(),
    })
    .with_input(
        "request.adapter_segments",
        Value::from_slice_i64(&segments, &[batch, 2])?,
    )
    .with_input(
        "request.adapter_counts",
        Value::from_slice_i64(&adapter_counts, &[batch])?,
    )
    .with_input(
        "request.adapter_scales",
        Value::from_slice_f32(&adapter_scales, &[batch, 2])?,
    )
    .with_input(
        "request.active",
        Value::from_raw_bytes(
            active.iter().map(|value| u8::from(*value)).collect(),
            &[batch],
            DataType::Bool,
        )?,
    )
    .with_input("activations", Value::from_slice_f32(values, &[batch, 2])?))
}

pub fn codec_request(waveform: &[f32]) -> anyhow::Result<PipelineGenerateRequest> {
    Ok(
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "request.waveform",
                Value::from_slice_f32(waveform, &[1, 1, i64::try_from(waveform.len())?])?,
            ),
    )
}

pub fn guided_diffusion_request(
    seeds: &[i64],
    prompts: &[i64],
) -> anyhow::Result<PipelineGenerateRequest> {
    ensure!(
        prompts.len() == seeds.len() * 2,
        "guided diffusion expects exactly two prompt ids per row"
    );
    let rows = i64::try_from(seeds.len())?;
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: options(3),
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(prompts, &[rows, 2])?,
    )
    .with_input(
        "request.negative_input_ids",
        Value::from_slice_i64(&vec![0; prompts.len()], &[rows, 2])?,
    )
    .with_input("request.seed", Value::from_slice_i64(seeds, &[rows])?)
    .with_input(
        "package.rng_offset",
        Value::from_slice_i64(&vec![0; seeds.len()], &[rows])?,
    )
    .with_input(
        "request.guidance_scale",
        Value::from_slice_f32(&vec![7.5; seeds.len()], &[rows])?,
    )
    .with_input(
        "package.false",
        Value::from_raw_bytes(vec![0; seeds.len()], &[rows], DataType::Bool)?,
    ))
}

pub fn masked_request(
    prompt_tokens: &[u32],
    masked_positions: &[bool],
    rng_offset: &[i64],
) -> anyhow::Result<PipelineGenerateRequest> {
    ensure!(
        rng_offset.len() == 1,
        "the hermetic masked fixture parity request is single-row"
    );
    ensure!(
        prompt_tokens.len() == masked_positions.len(),
        "masked prompt and mask lengths must match"
    );
    let sequence = i64::try_from(prompt_tokens.len())?;
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(prompt_tokens.to_vec()),
        options: options(3),
    })
    .with_input(
        "masked_positions",
        Value::from_raw_bytes(
            masked_positions
                .iter()
                .map(|value| u8::from(*value))
                .collect(),
            &[1, sequence],
            DataType::Bool,
        )?,
    )
    .with_input("rng_offset", Value::from_slice_i64(rng_offset, &[1])?))
}

pub fn tts_request(prompt_tokens: &[i64], batch: i64) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = usize::try_from(batch)?;
    ensure!(
        prompt_tokens.len() == rows * 2,
        "the hermetic TTS fixture expects exactly two prompt tokens per row"
    );
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![0]),
        options: options(1),
    })
    .with_input(
        "request.prompt_tokens",
        Value::from_slice_i64(prompt_tokens, &[batch, 2])?,
    )
    .with_input(
        "package.false",
        Value::from_raw_bytes(vec![0; rows], &[batch], DataType::Bool)?,
    )
    .with_input(
        "package.zero_batch",
        Value::from_slice_i64(&vec![0; rows], &[batch])?,
    )
    .with_input(
        "package.one_batch",
        Value::from_slice_i64(&vec![1; rows], &[batch])?,
    )
    .with_input(
        "package.true",
        Value::from_raw_bytes(vec![1; rows], &[batch], DataType::Bool)?,
    ))
}

pub fn video_request(latent_frames: i64, batch: i64) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = usize::try_from(batch)?;
    let elements = batch * latent_frames * 4 * 2 * 2;
    let noise: Vec<f32> = (0..elements)
        .map(|index| (index % 11) as f32 / 11.0 - 0.5)
        .collect();
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: options(3),
    })
    .with_input(
        "request.noise",
        Value::from_slice_f32(&noise, &[batch, latent_frames, 4, 2, 2])?,
    )
    .with_input(
        "request.encoder_hidden_states",
        Value::from_slice_f32(&vec![0.25; rows * 2 * 32], &[batch, 2, 32])?,
    )
    .with_input(
        "package.false",
        Value::from_raw_bytes(vec![0; rows], &[batch], DataType::Bool)?,
    ))
}
