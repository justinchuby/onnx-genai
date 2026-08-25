//! ORT decode-step drivers and position/step layout.
//!
//! Pure code motion from `decode.rs`.

use super::resolved_io::{shapes_compatible, validate_position_shape};
use super::state::DecodeState;
use super::values::{
    clone_value, empty_past_value, ensure_i64, is_gather_out_of_bounds, zero_state_value,
};
use super::*;

pub(crate) fn run_decode_session_logits(
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    align_runner_cursor(decode_state, token_ids, past_len)?;
    decode_state
        .runner
        .as_mut()
        .context("decode session runner not initialized")?
        .as_backend()
        .decode(token_ids, past_len)
        .map_err(map_decode_context_error)
}

/// Greedy fast-path sibling of [`run_decode_session_logits`]: advance the
/// runner one step and return only the argmax token id, or `None` if the runner
/// cannot select internally. Callers gate this on
/// [`DecodeState::runner_supports_argmax`], so `None` should not occur in
/// practice once the fast path is chosen.
pub(crate) fn run_decode_session_argmax(
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Option<u32>> {
    align_runner_cursor(decode_state, token_ids, past_len)?;
    decode_state
        .runner
        .as_mut()
        .context("decode session runner not initialized")?
        .as_backend()
        .decode_argmax(token_ids, past_len)
        .map_err(map_decode_context_error)
}

pub(crate) fn run_decode_session_sampled(
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
    params: &DeviceSampleParams,
) -> anyhow::Result<Option<u32>> {
    align_runner_cursor(decode_state, token_ids, past_len)?;
    decode_state
        .runner
        .as_mut()
        .context("decode session runner not initialized")?
        .as_backend()
        .decode_sampled(token_ids, past_len, params)
        .map_err(map_decode_context_error)
}

/// Align the runner's KV cursor to `past_len`, rewinding if it is ahead and
/// erroring if it is behind (replay is required). Shared by the logits and
/// argmax decode-session entry points.
fn align_runner_cursor(
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<()> {
    if token_ids.is_empty() {
        anyhow::bail!("decode session step requires at least one input token");
    }
    let current_len = decode_state.runner_len();
    if current_len > past_len {
        decode_state.rewind_runner(past_len)?;
    } else if current_len < past_len {
        anyhow::bail!(
            "decode session cursor {current_len} is behind requested past length {past_len}; replay is required"
        );
    }
    Ok(())
}

fn map_decode_context_error(error: anyhow::Error) -> anyhow::Error {
    let message = error.to_string();
    if is_gather_out_of_bounds(&message) {
        anyhow::anyhow!(
            "model context length exceeded during ORT decode; configure inference metadata `model.max_sequence_length` or GenerateOptions::max_context to stop cleanly before the context window is exceeded: {error}"
        )
    } else {
        error
    }
}

pub(crate) fn run_decode_step(
    session: &Session,
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Vec<Value>> {
    run_decode_step_with_extra(session, decode_state, token_ids, past_len, &[])
}

pub(crate) fn run_decode_step_with_extra(
    session: &Session,
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
    extra_inputs: &[(String, Value)],
) -> anyhow::Result<Vec<Value>> {
    if token_ids.is_empty() {
        anyhow::bail!("decode step requires at least one input token");
    }

    let seq_len = token_ids.len();
    let retained_past_len = decode_state.retained_kv_len(past_len);
    let (total_len, legacy_position_ids) =
        decode_step_layout(past_len, retained_past_len, seq_len)?;
    let input_ids = token_ids
        .iter()
        .map(|&id| i64::from(id))
        .collect::<Vec<_>>();
    let attention_mask = vec![1_i64; total_len];
    let mut position_step = if let Some(position_input) =
        decode_state.io.position_ids_input.as_deref()
    {
        let info = session
            .inputs()
            .iter()
            .find(|info| info.name == position_input)
            .with_context(|| {
                format!("declared position input '{position_input}' disappeared from graph inputs")
            })?;
        Some(build_position_step(
            info,
            decode_state.positions.as_ref(),
            decode_state.next_positions.as_deref(),
            past_len,
            seq_len,
            &legacy_position_ids,
            extra_inputs,
        )?)
    } else {
        None
    };

    let mut owned_inputs: Vec<(String, Value)> = Vec::new();
    for info in session.inputs() {
        if decode_state.io.is_token_input(&info.name) {
            owned_inputs.push((
                info.name.clone(),
                build_int_input(&input_ids, &[1, seq_len as i64], info)?,
            ));
        } else if decode_state.io.is_attention_mask_input(&info.name) {
            owned_inputs.push((
                info.name.clone(),
                build_int_input(&attention_mask, &[1, total_len as i64], info)?,
            ));
        } else if decode_state.io.is_position_ids_input(&info.name) {
            if position_step.is_none() {
                position_step = Some(build_position_step(
                    info,
                    decode_state.positions.as_ref(),
                    decode_state.next_positions.as_deref(),
                    past_len,
                    seq_len,
                    &legacy_position_ids,
                    extra_inputs,
                )?);
            }
            let step = position_step.as_ref().context(
                "position input was resolved without a generated or routed position tensor",
            )?;
            owned_inputs.push((info.name.clone(), clone_value(&step.value)?));
        } else if decode_state.use_kv && decode_state.kv_inputs.contains(&info.name) {
            let value = if retained_past_len == 0 {
                empty_past_value(info)?
            } else {
                clone_value(decode_state.past().get(&info.name).with_context(|| {
                    format!("missing cached KV tensor for input '{}'", info.name)
                })?)?
            };
            owned_inputs.push((info.name.clone(), value));
        } else if decode_state
            .io
            .state_pairs
            .iter()
            .any(|(input, _)| input == &info.name)
        {
            let value = match decode_state.loop_state.get(&info.name) {
                Some(value) => clone_value(value)?,
                None => zero_state_value(info)?,
            };
            owned_inputs.push((info.name.clone(), value));
        } else if let Some((_, value)) = extra_inputs.iter().find(|(name, _)| name == &info.name) {
            owned_inputs.push((info.name.clone(), clone_value(value)?));
        } else if decode_state.io.inputs_embeds_input.as_deref() == Some(info.name.as_str()) {
            anyhow::bail!(
                "declared inputs_embeds input '{}' was not supplied to the decode step; an embeds-driven decoder must receive its pre-embedded sequence via a pipeline dataflow edge",
                info.name
            );
        } else {
            anyhow::bail!(
                "unsupported model input '{}' with shape {:?}; declare its semantic role in pipeline.workflow.components.<component>.ports.roles or route it through pipeline metadata (declared state inputs: {:?})",
                info.name,
                info.shape,
                decode_state
                    .io
                    .state_pairs
                    .iter()
                    .map(|(input, _)| input)
                    .collect::<Vec<_>>()
            );
        }
    }

    let input_refs = owned_inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    let outputs = session.run(&input_refs).map_err(|e| {
        let message = e.to_string();
        if is_gather_out_of_bounds(&message) {
            anyhow::anyhow!(
                "model context length exceeded during ORT decode; configure inference metadata `model.max_sequence_length` or GenerateOptions::max_context to stop cleanly before the context window is exceeded: {e}"
            )
        } else {
            anyhow::anyhow!("ORT session run failed: {e}")
        }
    })?;

    if decode_state.use_kv {
        let mut next_past = HashMap::new();
        for (name, value) in session.output_names().iter().zip(outputs.iter()) {
            if let Some(past_name) = decode_state.present_to_past.get(name) {
                next_past.insert(past_name.clone(), clone_value(value)?);
            }
        }
        // The KV now covers every token processed so far; set the tensors and
        // their absolute length together so the state owns a length that cannot
        // drift from `past`. A windowed step then trims the physical rows in
        // place without changing this absolute count.
        decode_state.set_past(next_past, past_len + seq_len);
        decode_state.apply_window_after_step(session, past_len + seq_len, total_len)?;
    }
    if !decode_state.io.state_pairs.is_empty() {
        let mut replacements = HashMap::with_capacity(decode_state.io.state_pairs.len());
        for (input_name, output_name) in &decode_state.io.state_pairs {
            let output_index = session
                .output_names()
                .iter()
                .position(|name| name == output_name)
                .with_context(|| {
                    format!(
                        "declared loop-state output '{output_name}' disappeared from graph outputs"
                    )
                })?;
            let value = outputs.get(output_index).with_context(|| {
                format!("loop-state output '{output_name}' index was out of range")
            })?;
            let input_info = session
                .inputs()
                .iter()
                .find(|info| info.name == *input_name)
                .with_context(|| {
                    format!(
                        "declared loop-state input '{input_name}' disappeared from graph inputs"
                    )
                })?;
            if value.dtype() != input_info.dtype
                || !shapes_compatible(value.shape(), &input_info.shape)
            {
                anyhow::bail!(
                    "loop-state output '{output_name}' produced dtype {:?} shape {:?}, incompatible with next-step input '{input_name}' dtype {:?} shape {:?}",
                    value.dtype(),
                    value.shape(),
                    input_info.dtype,
                    input_info.shape
                );
            }
            replacements.insert(input_name.clone(), clone_value(value)?);
        }
        decode_state.loop_state = replacements;
    }
    if let Some(step) = position_step {
        decode_state.next_positions = Some(step.next);
    }

    Ok(outputs)
}

pub(super) fn decode_step_layout(
    absolute_past_len: usize,
    retained_past_len: usize,
    input_len: usize,
) -> anyhow::Result<(usize, Vec<i64>)> {
    let attended_len = retained_past_len
        .checked_add(input_len)
        .context("attention length overflow")?;
    let absolute_total_len = absolute_past_len
        .checked_add(input_len)
        .context("absolute position overflow")?;
    let position_ids = (absolute_past_len..absolute_total_len)
        .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((attended_len, position_ids))
}

pub(super) struct PositionStep {
    value: Value,
    next: Vec<i64>,
}

pub(super) fn build_position_step(
    info: &TensorInfo,
    program: Option<&PositionProgram>,
    next_positions: Option<&[i64]>,
    absolute_past_len: usize,
    input_len: usize,
    legacy_positions: &[i64],
    extra_inputs: &[(String, Value)],
) -> anyhow::Result<PositionStep> {
    ensure_i64(info)?;
    let rank = match program {
        Some(program) => program.rank,
        None if info.shape.len() == 2 => 1,
        None => {
            anyhow::bail!(
                "position input '{}' has shape {:?}; multi-axis position inputs require pipeline.positions metadata with an explicit rank",
                info.name,
                info.shape
            )
        }
    };
    validate_position_shape(info, rank)?;

    if let Some((_, supplied)) = extra_inputs.iter().find(|(name, _)| name == &info.name) {
        if supplied.dtype() != DataType::Int64 {
            anyhow::bail!(
                "routed position input '{}' must be Int64, got {:?}",
                info.name,
                supplied.dtype()
            );
        }
        validate_position_value_shape(info, supplied.shape(), rank, input_len)?;
        let data = supplied
            .to_vec_i64()
            .with_context(|| format!("failed to read routed position tensor '{}'", info.name))?;
        return Ok(PositionStep {
            next: next_position_axes(&data, rank, input_len)?,
            value: clone_value(supplied)?,
        });
    }

    let continuation = program
        .and_then(|program| program.continuation.as_deref())
        .unwrap_or("linear_increment");
    if continuation == "from_grid" && next_positions.is_none() {
        anyhow::bail!(
            "pipeline.positions continuation 'from_grid' requires the prefill position tensor '{}' to be supplied by a pipeline dataflow edge; route the processor-derived coordinates to that decoder input",
            info.name
        );
    }
    let absolute_start =
        i64::try_from(absolute_past_len).context("position id exceeds i64 range")?;
    let starts = if matches!(continuation, "carry_max" | "from_grid") {
        next_positions
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| vec![absolute_start; rank])
    } else {
        vec![absolute_start; rank]
    };
    if starts.len() != rank {
        anyhow::bail!(
            "position continuation for '{}' retained {} axes, but metadata declares rank {}",
            info.name,
            starts.len(),
            rank
        );
    }

    let (mut data, shape) = position_ids_from_starts(&starts, input_len)?;
    if rank == 1 && continuation == "linear_increment" {
        data.copy_from_slice(legacy_positions);
    }
    let next = starts
        .into_iter()
        .map(|start| {
            start
                .checked_add(i64::try_from(input_len).context("position length exceeds i64")?)
                .context("next position id overflow")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(PositionStep {
        value: Value::from_vec_i64(data, &shape)
            .with_context(|| format!("failed to build position input '{}'", info.name))?,
        next,
    })
}

/// Build the flat `i64` position-id values and physical tensor shape for one
/// decode step from the per-axis start positions. Each of the `starts.len()`
/// coordinate axes advances linearly across `input_len` sequence offsets. A
/// single axis yields a conventional `[1, input_len]` rank-2 tensor; multiple
/// axes yield a `[rank, 1, input_len]` rank-3 multi-axis (mrope) tensor.
///
/// Shared by the ORT step driver ([`build_position_step`]) and the native decode
/// drivers so both backends build byte-identical positions from one code path —
/// only the surrounding tensor type differs (`Value` vs native `Tensor`).
pub(crate) fn position_ids_from_starts(
    starts: &[i64],
    input_len: usize,
) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
    let rank = starts.len();
    let mut data = Vec::with_capacity(
        rank.checked_mul(input_len)
            .context("position tensor element count overflow")?,
    );
    for start in starts {
        for offset in 0..input_len {
            data.push(
                start
                    .checked_add(i64::try_from(offset).context("position offset exceeds i64")?)
                    .context("position id overflow")?,
            );
        }
    }
    let shape = if rank == 1 {
        vec![1, input_len as i64]
    } else {
        vec![rank as i64, 1, input_len as i64]
    };
    Ok((data, shape))
}

fn validate_position_value_shape(
    info: &TensorInfo,
    actual: &[i64],
    rank: usize,
    input_len: usize,
) -> anyhow::Result<()> {
    let expected = if rank == 1 {
        vec![1, input_len as i64]
    } else {
        vec![rank as i64, 1, input_len as i64]
    };
    if actual != expected {
        anyhow::bail!(
            "routed position input '{}' has shape {:?}, expected {:?} from pipeline.positions rank {} and decode sequence length {}",
            info.name,
            actual,
            expected,
            rank,
            input_len
        );
    }
    Ok(())
}

fn next_position_axes(data: &[i64], rank: usize, input_len: usize) -> anyhow::Result<Vec<i64>> {
    if data.len()
        != rank
            .checked_mul(input_len)
            .context("position tensor element count overflow")?
    {
        anyhow::bail!(
            "position tensor contains {} elements, expected {} axes × {} sequence positions",
            data.len(),
            rank,
            input_len
        );
    }
    data.chunks(input_len)
        .map(|axis| {
            axis.iter()
                .copied()
                .max()
                .context("position axis cannot be empty")?
                .checked_add(1)
                .context("next position id overflow")
        })
        .collect()
}

/// Build an integer graph input (token ids / attention mask) in the dtype the
/// graph declares. Most decoders take `Int64`, but encoder-decoder decoders such
/// as Whisper declare `Int32` `input_ids`; both are materialized from the same
/// `i64` host values so the caller stays dtype-agnostic.
fn build_int_input(values: &[i64], shape: &[i64], info: &TensorInfo) -> anyhow::Result<Value> {
    match info.dtype {
        DataType::Int64 => Value::from_slice_i64(values, shape)
            .map_err(|e| anyhow::anyhow!("failed to build Int64 input '{}': {e}", info.name)),
        DataType::Int32 => {
            let bytes = values
                .iter()
                .flat_map(|&value| (value as i32).to_le_bytes())
                .collect::<Vec<u8>>();
            Value::from_raw_bytes(bytes, shape, DataType::Int32)
                .map_err(|e| anyhow::anyhow!("failed to build Int32 input '{}': {e}", info.name))
        }
        other => anyhow::bail!(
            "input '{}' must be Int64 or Int32, got {other:?}",
            info.name
        ),
    }
}

#[cfg(test)]
mod position_helper_tests {
    use super::position_ids_from_starts;

    #[test]
    fn rank_one_builds_linear_row() {
        // A conventional decoder: single axis, one decode step at absolute
        // position 5 over 3 input positions → `[1, 3]` with `[5, 6, 7]`.
        let (data, shape) = position_ids_from_starts(&[5], 3).unwrap();
        assert_eq!(shape, vec![1, 3]);
        assert_eq!(data, vec![5, 6, 7]);
    }

    #[test]
    fn rank_three_single_token_replicates_across_axes() {
        // Rank-3 mrope, one token at absolute position 5: every coordinate axis
        // carries the same linear position → `[3, 1, 1]` with `[5, 5, 5]`.
        let (data, shape) = position_ids_from_starts(&[5, 5, 5], 1).unwrap();
        assert_eq!(shape, vec![3, 1, 1]);
        assert_eq!(data, vec![5, 5, 5]);
    }

    #[test]
    fn rank_three_multi_token_advances_each_axis() {
        // Rank-3 mrope prefill of 2 positions from absolute start 5: each axis
        // advances linearly → `[3, 1, 2]` with three `[5, 6]` streams.
        let (data, shape) = position_ids_from_starts(&[5, 5, 5], 2).unwrap();
        assert_eq!(shape, vec![3, 1, 2]);
        assert_eq!(data, vec![5, 6, 5, 6, 5, 6]);
    }
}
