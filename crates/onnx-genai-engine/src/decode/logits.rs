//! Extraction of next-token logits and hidden states from decode outputs.
//!
//! Pure code motion from `decode.rs`.

use super::*;

/// Locate the logits output selected by the resolved I/O contract.
fn logits_output_index(session: &Session, logits_output: Option<&str>) -> anyhow::Result<usize> {
    let declared = logits_output.context(
        "decoder logits role is unresolved; declare logits_output by giving the port the logits role in pipeline.workflow.components.<component>.ports.roles",
    )?;
    session
        .output_names()
        .iter()
        .position(|name| name == declared)
        .with_context(|| format!("declared logits output '{declared}' is not exposed by the graph"))
}

pub(crate) fn extract_next_token_logits_from_outputs(
    session: &Session,
    outputs: &[Value],
    logits_output: Option<&str>,
) -> anyhow::Result<Vec<f32>> {
    let logits_index = logits_output_index(session, logits_output)?;
    let logits = outputs
        .get(logits_index)
        .context("logits output index was out of range")?;
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {e}"))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(data),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }

        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {other:?}"),
    }
}

pub(super) fn extract_last_hidden(
    session: &Session,
    outputs: &[Value],
    output_name: &str,
) -> anyhow::Result<Vec<f32>> {
    let index = session
        .output_names()
        .iter()
        .position(|name| name == output_name)
        .with_context(|| {
            format!("target model did not expose hidden-state output '{output_name}'")
        })?;
    let value = outputs
        .get(index)
        .context("hidden-state output index was out of range")?;
    let shape = value.shape();
    let data = value
        .to_vec_f32_lossy()
        .map_err(|error| anyhow::anyhow!("Failed to read target hidden-state tensor: {error}"))?;
    match shape {
        [hidden] if *hidden > 0 => Ok(data),
        [seq, hidden] if *seq > 0 && *hidden > 0 => {
            let hidden = *hidden as usize;
            let start = (*seq as usize - 1) * hidden;
            Ok(data[start..start + hidden].to_vec())
        }
        [batch, seq, hidden] if *batch == 1 && *seq > 0 && *hidden > 0 => {
            let hidden = *hidden as usize;
            let start = (*seq as usize - 1) * hidden;
            Ok(data[start..start + hidden].to_vec())
        }
        [batch, seq, hc_mult, hidden] if *batch == 1 && *seq > 0 && *hc_mult > 0 && *hidden > 0 => {
            let state_width = (*hc_mult as usize)
                .checked_mul(*hidden as usize)
                .context("target HC state width overflow")?;
            let start = (*seq as usize - 1)
                .checked_mul(state_width)
                .context("target HC state offset overflow")?;
            Ok(data[start..start + state_width].to_vec())
        }
        other => anyhow::bail!(
            "unsupported target hidden-state tensor shape for '{output_name}': {other:?}"
        ),
    }
}

pub(crate) fn extract_logits_sequence_with_io(
    session: &Session,
    outputs: Vec<Value>,
    logits_output: Option<&str>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let logits_index = logits_output_index(session, logits_output)?;
    let logits = outputs
        .get(logits_index)
        .context("logits output index was out of range")?;
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {e}"))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(vec![data]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {other:?}"),
    }
}

pub(super) fn extract_logits_value_next(logits: &Value) -> anyhow::Result<Vec<f32>> {
    let sequence = extract_logits_value_sequence(logits)?;
    sequence
        .into_iter()
        .last()
        .context("logits tensor did not contain any sequence rows")
}

pub(super) fn extract_logits_value_sequence(logits: &Value) -> anyhow::Result<Vec<Vec<f32>>> {
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {e}"))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(vec![data]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {other:?}"),
    }
}
