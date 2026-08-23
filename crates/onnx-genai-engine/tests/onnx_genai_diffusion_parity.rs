//! Numeric parity between a Mobius-exported real Stable Diffusion workflow
//! package and the upstream `diffusers` pipeline it was built from.
//!
//! These tests execute the complete metadata path - text conditioning, latent
//! initialization, the denoiser loop with classifier-free guidance, the solver
//! and its carried history, the VAE decode - and compare every intermediate
//! against tensors recorded from `diffusers` at a pinned revision and seed.
//! They are skipped unless `MOBIUS_SD_DIFFUSION_PACKAGE_DIR` points at a
//! directory holding the exported packages and the recorded reference, because
//! real weights are far too large to check in.

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::{Path, PathBuf};

const PACKAGE_ENV: &str = "MOBIUS_SD_DIFFUSION_PACKAGE_DIR";

fn artifacts() -> Option<PathBuf> {
    std::env::var_os(PACKAGE_ENV).map(PathBuf::from)
}

/// Tensors recorded from the upstream pipeline, stored as raw little-endian
/// blobs plus a JSON index so the test needs no array container format.
struct Reference {
    directory: PathBuf,
    index: serde_json::Value,
}

impl Reference {
    fn load(root: &Path, name: &str) -> anyhow::Result<Self> {
        let directory = root.join("reference").join(name);
        let index = serde_json::from_slice(&std::fs::read(directory.join("index.json"))?)?;
        Ok(Self { directory, index })
    }

    fn entry(&self, name: &str) -> anyhow::Result<&serde_json::Value> {
        self.index
            .get("tensors")
            .and_then(|tensors| tensors.get(name))
            .ok_or_else(|| anyhow::anyhow!("reference has no tensor '{name}'"))
    }

    fn shape(&self, name: &str) -> anyhow::Result<Vec<i64>> {
        Ok(self
            .entry(name)?
            .get("shape")
            .and_then(|shape| shape.as_array())
            .ok_or_else(|| anyhow::anyhow!("reference tensor '{name}' has no shape"))?
            .iter()
            .map(|dimension| dimension.as_i64().unwrap_or_default())
            .collect())
    }

    fn bytes(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let file = self
            .entry(name)?
            .get("file")
            .and_then(|file| file.as_str())
            .ok_or_else(|| anyhow::anyhow!("reference tensor '{name}' has no file"))?;
        Ok(std::fs::read(self.directory.join(file))?)
    }

    fn f32(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self
            .bytes(name)?
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn i64(&self, name: &str) -> anyhow::Result<Vec<i64>> {
        Ok(self
            .bytes(name)?
            .chunks_exact(8)
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect())
    }

    fn value_f32(&self, name: &str) -> anyhow::Result<Value> {
        Ok(Value::from_slice_f32(&self.f32(name)?, &self.shape(name)?)?)
    }

    fn value_i64(&self, name: &str) -> anyhow::Result<Value> {
        Ok(Value::from_slice_i64(&self.i64(name)?, &self.shape(name)?)?)
    }

    /// One request row of a recorded rank-4 batch tensor.
    fn row_f32(&self, name: &str, row: usize) -> anyhow::Result<Value> {
        let shape = self.shape(name)?;
        let stride: usize = shape[1..]
            .iter()
            .map(|dimension| *dimension as usize)
            .product();
        let data = self.f32(name)?;
        let mut row_shape = shape.clone();
        row_shape[0] = 1;
        Ok(Value::from_slice_f32(
            &data[row * stride..(row + 1) * stride],
            &row_shape,
        )?)
    }

    fn row_i64(&self, name: &str, row: usize) -> anyhow::Result<Value> {
        let shape = self.shape(name)?;
        let stride: usize = shape[1..]
            .iter()
            .map(|dimension| *dimension as usize)
            .product();
        let data = self.i64(name)?;
        let mut row_shape = shape.clone();
        row_shape[0] = 1;
        Ok(Value::from_slice_i64(
            &data[row * stride..(row + 1) * stride],
            &row_shape,
        )?)
    }
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> anyhow::Result<f32> {
    anyhow::ensure!(
        left.len() == right.len(),
        "tensor element counts differ: {} vs {}",
        left.len(),
        right.len()
    );
    Ok(left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max))
}

/// float32 agreement budget. The absolute term matches the tolerance Mobius
/// uses for float32 integration parity; the relative term accounts for the
/// dynamic range of a latent, which starts at `sigma_max` (about 60 here) and
/// therefore carries proportionally larger rounding error than a noise
/// prediction of order 1.
const PARITY_ATOL: f32 = 1e-4;
const PARITY_RTOL: f32 = 1e-4;

fn assert_close(label: &str, got: &[f32], want: &[f32]) -> anyhow::Result<()> {
    let difference = max_abs_diff(got, want)?;
    let magnitude = want
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()));
    let tolerance = PARITY_ATOL + PARITY_RTOL * magnitude;
    anyhow::ensure!(
        difference <= tolerance,
        "{label} differs from the diffusers reference by {difference:e} \
         (tolerance {tolerance:e} for reference magnitude {magnitude:e})"
    );
    println!("{label}: max|diff| = {difference:e} (tolerance {tolerance:e})");
    Ok(())
}

fn assert_identical(label: &str, got: &[f32], want: &[f32]) -> anyhow::Result<()> {
    let difference = max_abs_diff(got, want)?;
    anyhow::ensure!(
        difference == 0.0,
        "{label} is not bit-identical: {difference:e}"
    );
    println!("{label}: identical");
    Ok(())
}

/// Compare a trajectory the workflow appended along its last axis against the
/// recorded per-step tensors. `append` concatenates on the final dimension, so
/// step `s` of row `b` lives at `[b, c, h, s * width + w]`.
fn assert_trajectory(
    label: &str,
    emitted: &Value,
    reference: &Reference,
    name: &str,
) -> anyhow::Result<()> {
    let recorded_shape = reference.shape(name)?;
    let (steps, batch, channels, height, width) = (
        recorded_shape[0] as usize,
        recorded_shape[1] as usize,
        recorded_shape[2] as usize,
        recorded_shape[3] as usize,
        recorded_shape[4] as usize,
    );
    let emitted_shape = emitted.shape().to_vec();
    anyhow::ensure!(
        emitted_shape
            == vec![
                batch as i64,
                channels as i64,
                height as i64,
                (steps * width) as i64
            ],
        "{label} has shape {emitted_shape:?}, expected {:?}",
        vec![batch, channels, height, steps * width]
    );
    let emitted = emitted.to_vec_f32()?;
    let recorded = reference.f32(name)?;
    let mut reordered = vec![0.0f32; recorded.len()];
    for step in 0..steps {
        for row in 0..batch {
            for channel in 0..channels {
                for line in 0..height {
                    for column in 0..width {
                        let source = ((row * channels + channel) * height + line) * (steps * width)
                            + step * width
                            + column;
                        let target = (((step * batch + row) * channels + channel) * height + line)
                            * width
                            + column;
                        reordered[target] = emitted[source];
                    }
                }
            }
        }
    }
    assert_close(label, &reordered, &recorded)
}

fn options(steps: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: steps,
        ..Default::default()
    }
}

fn guided_request(
    reference: &Reference,
    batch: i64,
    guidance_scale: f32,
    steps: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = usize::try_from(batch)?;
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: options(steps),
    })
    .with_input("request.input_ids", reference.value_i64("cond_input_ids")?)
    .with_input(
        "request.negative_input_ids",
        reference.value_i64("uncond_input_ids")?,
    )
    .with_input("request.noise", reference.value_f32("noise")?)
    .with_input(
        "request.guidance_scale",
        Value::from_slice_f32(&vec![guidance_scale; rows], &[batch])?,
    )
    .with_input(
        "package.false",
        Value::from_raw_bytes(vec![0; rows], &[batch], DataType::Bool)?,
    ))
}

fn run_parity(package: &str, baseline: &str, batch: i64) -> anyhow::Result<()> {
    let Some(root) = artifacts() else {
        eprintln!("skipping: {PACKAGE_ENV} is not set");
        return Ok(());
    };
    let reference = Reference::load(&root, baseline)?;
    let steps = reference.shape("timesteps")?[0] as usize;
    let guidance_scale = reference.index["config"]["guidance_scale"]
        .as_f64()
        .unwrap_or(7.5) as f32;
    let mut engine = Engine::from_dir(&root.join(package), EngineConfig::default())?;
    let output =
        engine.run_pipeline_outputs(guided_request(&reference, batch, guidance_scale, steps)?)?;

    assert_eq!(output["latent"].shape(), reference.shape("final_latent")?);
    assert_eq!(output["image"].shape(), reference.shape("decoded")?);
    assert_trajectory(
        &format!("{package} noise_estimate"),
        &output["noise_estimate"],
        &reference,
        "noise_pred_guided",
    )?;
    assert_trajectory(
        &format!("{package} latent_trajectory"),
        &output["latent_trajectory"],
        &reference,
        "latent_trajectory",
    )?;
    assert_close(
        &format!("{package} latent"),
        &output["latent"].to_vec_f32()?,
        &reference.f32("final_latent")?,
    )?;
    assert_close(
        &format!("{package} image"),
        &output["image"].to_vec_f32()?,
        &reference.f32("decoded")?,
    )?;
    Ok(())
}

#[test]
fn real_stable_diffusion_euler_guided_workflow_matches_diffusers() -> anyhow::Result<()> {
    run_parity("package_euler_cfg", "baseline_euler_b1", 1)
}

#[test]
fn real_stable_diffusion_euler_batch_two_matches_diffusers() -> anyhow::Result<()> {
    run_parity("package_euler_cfg", "baseline_euler_b2", 2)
}

#[test]
fn real_stable_diffusion_multistep_batch_two_matches_diffusers() -> anyhow::Result<()> {
    run_parity("package_dpmpp_cfg", "baseline_dpmpp_b2", 2)
}

/// A batched run must produce exactly what each row produces on its own: the
/// solver state, the carried history, and the conditioning are per row, so no
/// row may observe another row's prompt or noise.
#[test]
fn real_stable_diffusion_batched_rows_are_independent() -> anyhow::Result<()> {
    let Some(root) = artifacts() else {
        eprintln!("skipping: {PACKAGE_ENV} is not set");
        return Ok(());
    };
    for (package, baseline) in [
        ("package_euler_cfg", "baseline_euler_b2"),
        ("package_dpmpp_cfg", "baseline_dpmpp_b2"),
    ] {
        let reference = Reference::load(&root, baseline)?;
        let steps = reference.shape("timesteps")?[0] as usize;
        let mut engine = Engine::from_dir(&root.join(package), EngineConfig::default())?;
        let batched =
            engine.run_pipeline_outputs(guided_request(&reference, 2, 7.5, steps)?)?["latent"]
                .to_vec_f32()?;
        let row_elements = batched.len() / 2;
        for row in 0..2 {
            let request = PipelineGenerateRequest::new(GenerateRequest {
                prompt: GeneratePrompt::TokenIds(vec![]),
                options: options(steps),
            })
            .with_input(
                "request.input_ids",
                reference.row_i64("cond_input_ids", row)?,
            )
            .with_input(
                "request.negative_input_ids",
                reference.row_i64("uncond_input_ids", row)?,
            )
            .with_input("request.noise", reference.row_f32("noise", row)?)
            .with_input(
                "request.guidance_scale",
                Value::from_slice_f32(&[7.5], &[1])?,
            )
            .with_input(
                "package.false",
                Value::from_raw_bytes(vec![0], &[1], DataType::Bool)?,
            );
            let single = engine.run_pipeline_outputs(request)?["latent"].to_vec_f32()?;
            assert_close(
                &format!("{package} row {row} latent"),
                &single,
                &batched[row * row_elements..(row + 1) * row_elements],
            )?;
        }
    }
    Ok(())
}

/// The seeded package draws its own latent from counter RNG state carried in
/// the metadata. The draw must be reproducible, private to each row, and it
/// must advance the counter the workflow returns.
#[test]
fn real_stable_diffusion_seeded_latents_are_deterministic_per_row() -> anyhow::Result<()> {
    let Some(root) = artifacts() else {
        eprintln!("skipping: {PACKAGE_ENV} is not set");
        return Ok(());
    };
    let mut engine = Engine::from_dir(&root.join("package_euler_seeded"), EngineConfig::default())?;
    let reference = Reference::load(&root, "baseline_euler_b2")?;
    let recorded_cond = reference.i64("cond_input_ids")?;
    let recorded_uncond = reference.i64("uncond_input_ids")?;
    let width = reference.shape("cond_input_ids")?[1] as usize;
    // Each row is a (seed, recorded prompt) pair so a row can be moved between
    // batch positions without changing anything the row itself observes.
    let seeded = |rows: &[(i64, usize)]| -> anyhow::Result<PipelineGenerateRequest> {
        let batch = i64::try_from(rows.len())?;
        let mut seeds = Vec::new();
        let mut tokens = Vec::new();
        let mut negative = Vec::new();
        for (seed, prompt) in rows {
            seeds.push(*seed);
            tokens.extend_from_slice(&recorded_cond[prompt * width..(prompt + 1) * width]);
            negative.extend_from_slice(&recorded_uncond[prompt * width..(prompt + 1) * width]);
        }
        Ok(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![]),
            options: options(5),
        })
        .with_input(
            "request.input_ids",
            Value::from_slice_i64(&tokens, &[batch, width as i64])?,
        )
        .with_input(
            "request.negative_input_ids",
            Value::from_slice_i64(&negative, &[batch, width as i64])?,
        )
        .with_input("request.seed", Value::from_slice_i64(&seeds, &[batch])?)
        .with_input(
            "package.rng_offset",
            Value::from_slice_i64(&vec![0; rows.len()], &[batch])?,
        )
        .with_input(
            "request.guidance_scale",
            Value::from_slice_f32(&vec![7.5; rows.len()], &[batch])?,
        )
        .with_input(
            "package.false",
            Value::from_raw_bytes(vec![0; rows.len()], &[batch], DataType::Bool)?,
        ))
    };

    let first = engine.run_pipeline_outputs(seeded(&[(1234, 0), (4321, 1)])?)?;
    let repeat = engine.run_pipeline_outputs(seeded(&[(1234, 0), (4321, 1)])?)?;
    let latent = first["latent"].to_vec_f32()?;
    assert_identical(
        "seeded latent is reproducible",
        &latent,
        &repeat["latent"].to_vec_f32()?,
    )?;
    // The workflow returns the advanced counter so a follow-up request can
    // continue the same stream instead of redrawing it.
    assert_eq!(first["rng_offset"].to_vec_i64()?, vec![1, 1]);

    let row_elements = latent.len() / 2;
    let other_seed = engine.run_pipeline_outputs(seeded(&[(4321, 0)])?)?["latent"].to_vec_f32()?;
    let difference = max_abs_diff(&other_seed, &latent[..row_elements])?;
    anyhow::ensure!(
        difference > 1e-3,
        "seed 4321 must not reproduce seed 1234's latent for the same prompt"
    );

    // Moving a row to a different batch position must not change its draw.
    let swapped =
        engine.run_pipeline_outputs(seeded(&[(4321, 1), (1234, 0)])?)?["latent"].to_vec_f32()?;
    assert_close(
        "row moved from batch position 1 to 0",
        &swapped[..row_elements],
        &latent[row_elements..],
    )?;
    assert_close(
        "row moved from batch position 0 to 1",
        &swapped[row_elements..],
        &latent[..row_elements],
    )?;

    // A single-row request must reproduce the row it was taken from.
    let alone = engine.run_pipeline_outputs(seeded(&[(4321, 1)])?)?["latent"].to_vec_f32()?;
    assert_close("row 1 run alone", &alone, &latent[row_elements..])?;

    // A non-zero starting counter must move the stream somewhere else.
    let advanced = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: options(5),
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(&recorded_cond[..width], &[1, width as i64])?,
    )
    .with_input(
        "request.negative_input_ids",
        Value::from_slice_i64(&recorded_uncond[..width], &[1, width as i64])?,
    )
    .with_input("request.seed", Value::from_slice_i64(&[1234], &[1])?)
    .with_input("package.rng_offset", Value::from_slice_i64(&[1], &[1])?)
    .with_input(
        "request.guidance_scale",
        Value::from_slice_f32(&[7.5], &[1])?,
    )
    .with_input(
        "package.false",
        Value::from_raw_bytes(vec![0], &[1], DataType::Bool)?,
    );
    let advanced = engine.run_pipeline_outputs(advanced)?;
    anyhow::ensure!(
        max_abs_diff(&advanced["latent"].to_vec_f32()?, &latent[..row_elements])? > 1e-3,
        "advancing the RNG counter must draw a different latent"
    );
    assert_eq!(advanced["rng_offset"].to_vec_i64()?, vec![2]);
    Ok(())
}
