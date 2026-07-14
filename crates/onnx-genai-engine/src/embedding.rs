//! Single-pass hidden-state embedding inference and pooling.

use crate::decode::{DecodeState, run_decode_step};
use crate::{Engine, TokenId};
use anyhow::Context;
use onnx_genai_ort::{DataType, Session, Value};

/// Pooling strategy applied to per-token hidden states.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EmbeddingPooling {
    /// Average every token position in the input sequence.
    #[default]
    Mean,
    /// Use the final token position's hidden state.
    LastToken,
}

/// Options controlling hidden-state embedding extraction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingOptions {
    /// How per-token hidden states are reduced to one vector.
    pub pooling: EmbeddingPooling,
    /// Whether to L2-normalize the pooled vector.
    pub normalize: bool,
    /// Explicit hidden-state output name, or `None` to auto-detect it.
    pub hidden_state_output: Option<String>,
}

impl Engine {
    /// Produce a mean-pooled embedding without normalization.
    pub fn embed(&mut self, input_ids: &[TokenId]) -> anyhow::Result<Vec<f32>> {
        self.embed_with_options(input_ids, EmbeddingOptions::default())
    }

    /// Tokenize `text` with the model's tokenizer, then produce a mean-pooled
    /// embedding without normalization.
    pub fn embed_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.embed_text_with_options(text, EmbeddingOptions::default())
    }

    /// Tokenize `text` with the model's tokenizer, then produce a pooled
    /// embedding from the model's per-token hidden-state output.
    pub fn embed_text_with_options(
        &mut self,
        text: &str,
        options: EmbeddingOptions,
    ) -> anyhow::Result<Vec<f32>> {
        let input_ids = self.tokenize(text)?;
        self.embed_with_options(&input_ids, options)
    }

    /// Produce a pooled embedding from the model's per-token hidden-state output.
    pub fn embed_with_options(
        &mut self,
        input_ids: &[TokenId],
        options: EmbeddingOptions,
    ) -> anyhow::Result<Vec<f32>> {
        if input_ids.is_empty() {
            anyhow::bail!("embedding input must contain at least one token");
        }

        let hidden_output =
            resolve_hidden_state_output(&self.session, options.hidden_state_output.as_deref())?
                .to_string();
        let mut decode_state = DecodeState::new(&self.session)
            .context("failed to initialize embedding model inputs")?;
        let outputs = run_decode_step(&self.session, &mut decode_state, input_ids, 0)
            .context("embedding model forward pass failed")?;
        let hidden =
            extract_hidden_sequence(&self.session, &outputs, &hidden_output, input_ids.len())?;
        pool_hidden_states(
            &hidden.data,
            hidden.positions,
            hidden.hidden_size,
            options.pooling,
            options.normalize,
        )
    }
}

struct HiddenSequence {
    data: Vec<f32>,
    positions: usize,
    hidden_size: usize,
}

fn resolve_hidden_state_output<'a>(
    session: &'a Session,
    requested: Option<&str>,
) -> anyhow::Result<&'a str> {
    if let Some(requested) = requested {
        let output = session
            .outputs()
            .iter()
            .find(|output| output.name == requested)
            .with_context(|| {
                format!(
                    "model does not expose requested hidden-state output '{requested}'; available outputs: {:?}",
                    session.output_names()
                )
            })?;
        validate_hidden_output(output)?;
        return Ok(&output.name);
    }

    let candidates = session
        .outputs()
        .iter()
        .filter(|output| {
            output.name.to_ascii_lowercase().contains("hidden")
                && validate_hidden_output(output).is_ok()
        })
        .collect::<Vec<_>>();

    for preferred in ["last_hidden_state", "hidden_states"] {
        if let Some(output) = candidates
            .iter()
            .find(|output| output.name.eq_ignore_ascii_case(preferred))
        {
            return Ok(&output.name);
        }
    }

    let mut numbered = candidates
        .iter()
        .filter_map(|output| {
            output
                .name
                .to_ascii_lowercase()
                .strip_prefix("hidden_states.")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .map(|layer| (layer, output))
        })
        .collect::<Vec<_>>();
    numbered.sort_by_key(|(layer, _)| *layer);
    if let Some((_, output)) = numbered.last() {
        return Ok(&output.name);
    }

    if let [output] = candidates.as_slice() {
        return Ok(&output.name);
    }
    if candidates.is_empty() {
        anyhow::bail!(
            "model does not expose a usable per-token hidden-state output; available outputs: {:?}",
            session.output_names()
        );
    }
    anyhow::bail!(
        "model exposes multiple hidden-state outputs {:?}; set EmbeddingOptions::hidden_state_output explicitly",
        candidates
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>()
    )
}

fn validate_hidden_output(output: &onnx_genai_ort::TensorInfo) -> anyhow::Result<()> {
    if !matches!(output.dtype, DataType::Float32 | DataType::Float16) {
        anyhow::bail!(
            "hidden-state output '{}' must be Float32 or Float16, got {:?}",
            output.name,
            output.dtype
        );
    }
    if !matches!(output.shape.len(), 1..=3) {
        anyhow::bail!(
            "hidden-state output '{}' must have rank 1, 2, or 3, got shape {:?}",
            output.name,
            output.shape
        );
    }
    Ok(())
}

fn extract_hidden_sequence(
    session: &Session,
    outputs: &[Value],
    output_name: &str,
    input_len: usize,
) -> anyhow::Result<HiddenSequence> {
    let index = session
        .output_names()
        .iter()
        .position(|name| name == output_name)
        .with_context(|| format!("model did not return hidden-state output '{output_name}'"))?;
    let value = outputs
        .get(index)
        .context("hidden-state output index was out of range")?;
    let shape = value.shape();
    let data = value
        .to_vec_f32_lossy()
        .map_err(|error| anyhow::anyhow!("failed to read hidden-state output: {error}"))?;

    let (positions, hidden_size) = match shape {
        [hidden] if input_len == 1 && *hidden > 0 => (1, *hidden as usize),
        [positions, hidden] if *positions == input_len as i64 && *positions > 0 && *hidden > 0 => {
            (*positions as usize, *hidden as usize)
        }
        [batch, positions, hidden]
            if *batch == 1 && *positions == input_len as i64 && *positions > 0 && *hidden > 0 =>
        {
            (*positions as usize, *hidden as usize)
        }
        other => anyhow::bail!(
            "hidden-state output '{output_name}' must contain one row per input token; input length is {input_len}, output shape is {:?}",
            other
        ),
    };
    if data.len() != positions * hidden_size {
        anyhow::bail!(
            "hidden-state output '{output_name}' contains {} values, expected {} positions * {} hidden dimensions",
            data.len(),
            positions,
            hidden_size
        );
    }
    Ok(HiddenSequence {
        data,
        positions,
        hidden_size,
    })
}

fn pool_hidden_states(
    hidden: &[f32],
    positions: usize,
    hidden_size: usize,
    pooling: EmbeddingPooling,
    normalize: bool,
) -> anyhow::Result<Vec<f32>> {
    if positions == 0 || hidden_size == 0 || hidden.len() != positions * hidden_size {
        anyhow::bail!(
            "invalid hidden-state matrix: {} values for {positions} positions and hidden size {hidden_size}",
            hidden.len()
        );
    }

    let mut pooled = match pooling {
        EmbeddingPooling::Mean => {
            let mut pooled = vec![0.0f32; hidden_size];
            for row in hidden.chunks_exact(hidden_size) {
                for (pooled, value) in pooled.iter_mut().zip(row) {
                    *pooled += value;
                }
            }
            let scale = 1.0 / positions as f32;
            for value in &mut pooled {
                *value *= scale;
            }
            pooled
        }
        EmbeddingPooling::LastToken => hidden[(positions - 1) * hidden_size..].to_vec(),
    };

    if normalize {
        let norm = pooled
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            let inverse = (1.0 / norm) as f32;
            for value in &mut pooled {
                *value *= inverse;
            }
        }
    }
    Ok(pooled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineConfig;
    use onnx_genai_ort::SessionOptions;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard};

    fn model_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn fixture(name: &str) -> anyhow::Result<PathBuf> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
            .canonicalize()?)
    }

    fn engine(name: &str) -> anyhow::Result<Engine> {
        Engine::from_dir_with_session_options(
            &fixture(name)?,
            EngineConfig::default(),
            SessionOptions::default().with_intra_op_threads(1),
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "embedding[{index}] was {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn pools_synthetic_hidden_states_exactly() -> anyhow::Result<()> {
        let hidden = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        assert_eq!(
            pool_hidden_states(&hidden, 2, 3, EmbeddingPooling::Mean, false)?,
            vec![3.0, 5.0, 8.0]
        );
        assert_eq!(
            pool_hidden_states(&hidden, 2, 3, EmbeddingPooling::LastToken, false)?,
            vec![5.0, 8.0, 13.0]
        );
        assert_close(
            &pool_hidden_states(&hidden, 2, 3, EmbeddingPooling::LastToken, true)?,
            &[
                5.0 / 258.0f32.sqrt(),
                8.0 / 258.0f32.sqrt(),
                13.0 / 258.0f32.sqrt(),
            ],
        );
        Ok(())
    }

    #[test]
    fn pools_fixture_hidden_states_with_mean_last_and_normalization() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let mut engine = engine("tiny-mtp-full")?;
        let input_ids = [2, 4, 3];
        let output_name = resolve_hidden_state_output(&engine.session, None)?.to_string();
        let mut decode_state = DecodeState::new(&engine.session)?;
        let outputs = run_decode_step(&engine.session, &mut decode_state, &input_ids, 0)?;
        let hidden =
            extract_hidden_sequence(&engine.session, &outputs, &output_name, input_ids.len())?;

        let mut expected_mean = vec![0.0f32; hidden.hidden_size];
        for row in hidden.data.chunks_exact(hidden.hidden_size) {
            for (mean, value) in expected_mean.iter_mut().zip(row) {
                *mean += value;
            }
        }
        for value in &mut expected_mean {
            *value /= hidden.positions as f32;
        }
        let expected_last = hidden.data[(hidden.positions - 1) * hidden.hidden_size..].to_vec();

        let mean = engine.embed_with_options(
            &input_ids,
            EmbeddingOptions {
                pooling: EmbeddingPooling::Mean,
                ..Default::default()
            },
        )?;
        let last = engine.embed_with_options(
            &input_ids,
            EmbeddingOptions {
                pooling: EmbeddingPooling::LastToken,
                ..Default::default()
            },
        )?;
        assert_close(&mean, &expected_mean);
        assert_close(&last, &expected_last);

        let normalized = engine.embed_with_options(
            &input_ids,
            EmbeddingOptions {
                pooling: EmbeddingPooling::Mean,
                normalize: true,
                hidden_state_output: None,
            },
        )?;
        let norm = normalized
            .iter()
            .map(|&value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() <= f32::EPSILON * 4.0, "{norm}");
        Ok(())
    }

    #[test]
    fn logits_only_model_returns_a_clear_capability_error() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let mut engine = engine("tiny-llm")?;
        let error = engine.embed(&[2, 4, 3]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not expose a usable per-token hidden-state output"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn rejects_empty_embedding_input() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let mut engine = engine("tiny-mtp-full")?;
        assert_eq!(
            engine.embed(&[]).unwrap_err().to_string(),
            "embedding input must contain at least one token"
        );
        Ok(())
    }

    #[test]
    fn tokenize_round_trips_and_matches_internal_path() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let engine = engine("tiny-mtp-full")?;
        let ids = engine.tokenize("hello world")?;
        assert!(!ids.is_empty(), "tokenizer produced no ids");
        // The public seam must agree with the tokenizer path the engine owns.
        let expected = engine.tokenizer.encode("hello world")?;
        assert_eq!(ids, expected);
        Ok(())
    }

    #[test]
    fn embed_text_agrees_with_tokenize_then_embed() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let mut engine = engine("tiny-mtp-full")?;
        let text = "hello world";
        let ids = engine.tokenize(text)?;

        let options = EmbeddingOptions {
            pooling: EmbeddingPooling::Mean,
            normalize: true,
            hidden_state_output: None,
        };
        let via_ids = engine.embed_with_options(&ids, options.clone())?;
        let via_text = engine.embed_text_with_options(text, options)?;
        assert_close(&via_text, &via_ids);

        let default_via_ids = engine.embed(&ids)?;
        let default_via_text = engine.embed_text(text)?;
        assert_close(&default_via_text, &default_via_ids);
        Ok(())
    }
}
