//! Resolved decode-step graph I/O port bindings.
//!
//! Pure code motion from `decode.rs`: I/O resolution for the decode step.

use super::values::{ensure_i64, is_token_input_name};
use super::*;

/// Resolved graph I/O port bindings for the decode step.
///
/// Built from an explicit metadata `io` block when a model package declares one
/// (via [`ModelIoSpec`]), or derived from historical tensor-name conventions
/// otherwise. When [`ResolvedIo::explicit`] is `false`, the scalar port fields
/// are `None` and the decode step falls back to tensor-name conventions.
///
/// TRANSITIONAL: the convention fallback exists only until every model package
/// emits an `io` block. Phase 2 removes the fallback, at which point `explicit`
/// is always `true` and the `is_*` helpers collapse to direct name comparisons.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResolvedIo {
    /// True when built from an explicit metadata `io` block.
    pub(super) explicit: bool,
    pub(crate) token_input: Option<String>,
    pub(crate) inputs_embeds_input: Option<String>,
    pub(crate) attention_mask_input: Option<String>,
    pub(crate) position_ids_input: Option<String>,
    pub(crate) logits_output: Option<String>,
    pub(crate) hidden_output: Option<String>,
    /// `(past_input, present_output)` pairs, positionally paired. Empty for a
    /// non-KV graph.
    pub(crate) kv_pairs: Vec<(String, String)>,
    /// Fixed loop-carried `(input, output)` pairs with replace semantics.
    pub(crate) state_pairs: Vec<(String, String)>,
    /// Encoder-decoder cross-attention `(decoder_input, encoder_output)` pairs.
    ///
    /// The `input` is the decoder's `past_*_cross_%d` graph input; the `output`
    /// names the ENCODER graph output (`present_*_cross_%d`) that produces the
    /// value. The encoder runs once as a prompt-phase prologue and its cross-KV
    /// outputs are STATIC for the whole decode: they encode the full audio/text
    /// prompt and never grow or change across autoregressive steps, so the
    /// pipeline binds them once and re-supplies the same tensors every step.
    /// The output side is therefore intentionally NOT validated against the
    /// decoder graph here (it is an encoder port, resolved from the shared pool
    /// by the pipeline).
    pub(crate) cross_kv_pairs: Vec<(String, String)>,
}

fn resolve_state_pairs(
    session: &Session,
    declared: Option<&[LoopStatePair]>,
    kv_pairs: &[(String, String)],
) -> anyhow::Result<Vec<(String, String)>> {
    let Some(declared) = declared else {
        return Ok(Vec::new());
    };
    let kv_inputs = kv_pairs
        .iter()
        .map(|(input, _)| input.as_str())
        .collect::<HashSet<_>>();
    let kv_outputs = kv_pairs
        .iter()
        .map(|(_, output)| output.as_str())
        .collect::<HashSet<_>>();
    let mut inputs = HashSet::new();
    let mut outputs = HashSet::new();
    let mut resolved = Vec::with_capacity(declared.len());

    for pair in declared {
        let init = pair.init.as_deref().unwrap_or("zeros");
        if init != "zeros" {
            anyhow::bail!(
                "state pair '{}'=>'{}' declares unsupported init '{init}'; supported initializers: zeros",
                pair.input,
                pair.output
            );
        }
        let update = pair.update.as_deref().unwrap_or("replace");
        if update != "replace" {
            anyhow::bail!(
                "state pair '{}'=>'{}' declares unsupported update '{update}'; supported updates: replace",
                pair.input,
                pair.output
            );
        }
        if !inputs.insert(pair.input.as_str()) {
            anyhow::bail!("state_pairs declares input '{}' more than once", pair.input);
        }
        if !outputs.insert(pair.output.as_str()) {
            anyhow::bail!(
                "state_pairs declares output '{}' more than once",
                pair.output
            );
        }
        if kv_inputs.contains(pair.input.as_str())
            || kv_outputs.contains(pair.input.as_str())
            || kv_inputs.contains(pair.output.as_str())
            || kv_outputs.contains(pair.output.as_str())
        {
            anyhow::bail!(
                "state pair '{}'=>'{}' overlaps declared KV ports; fixed replace-state and KV cache ports must be separate",
                pair.input,
                pair.output
            );
        }
        let input = session
            .inputs()
            .iter()
            .find(|info| info.name == pair.input)
            .with_context(|| {
                format!(
                    "state_pairs declares input '{}' but the graph does not expose it; graph inputs: {:?}",
                    pair.input,
                    session.input_names()
                )
            })?;
        let output = session
            .outputs()
            .iter()
            .find(|info| info.name == pair.output)
            .with_context(|| {
                format!(
                    "state_pairs declares output '{}' but the graph does not expose it; graph outputs: {:?}",
                    pair.output,
                    session.output_names()
                )
            })?;
        if input.dtype != output.dtype {
            anyhow::bail!(
                "state pair '{}'=>'{}' has incompatible dtypes: input {:?}, output {:?}; replace-state ports must match",
                pair.input,
                pair.output,
                input.dtype,
                output.dtype
            );
        }
        if !shapes_compatible(&input.shape, &output.shape) {
            anyhow::bail!(
                "state pair '{}'=>'{}' has incompatible shapes: input {:?}, output {:?}; replace-state ports must match",
                pair.input,
                pair.output,
                input.shape,
                output.shape
            );
        }
        if input.shape.iter().any(|dimension| *dimension <= 0) {
            anyhow::bail!(
                "state input '{}' has dynamic or invalid shape {:?}; zero initialization requires every fixed-state dimension to be concrete and positive",
                pair.input,
                input.shape
            );
        }
        if !matches!(
            input.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16 | DataType::Int64
        ) {
            anyhow::bail!(
                "state input '{}' has unsupported zero-initialization dtype {:?}; supported dtypes: Float32, Float16, BFloat16, Int64",
                pair.input,
                input.dtype
            );
        }
        resolved.push((pair.input.clone(), pair.output.clone()));
    }

    Ok(resolved)
}

/// Resolve encoder-decoder cross-attention KV bindings into
/// `(decoder_input, encoder_output)` pairs.
///
/// `inputs` are the decoder's `past_*_cross_%d` graph inputs and are validated
/// against the decoder graph. `outputs` name the ENCODER graph outputs
/// (`present_*_cross_%d`) that supply each value; they are deliberately NOT
/// validated against the decoder graph here because they belong to a different
/// component. The pipeline resolves them from the shared tensor pool after the
/// encoder prologue and binds the resulting tensors as static decoder inputs on
/// every step. Cross-KV is computed once from the whole prompt and never
/// changes across decode steps, which is why it is carried separately from the
/// growing self-attention `kv_pairs`.
fn resolve_cross_kv_pairs(
    session: &Session,
    inputs: Option<&[String]>,
    outputs: Option<&[String]>,
) -> anyhow::Result<Vec<(String, String)>> {
    match (inputs, outputs) {
        (Some(inputs), Some(outputs)) => {
            if inputs.len() != outputs.len() {
                anyhow::bail!(
                    "io.cross_kv_inputs ({}) and io.cross_kv_outputs ({}) must have equal length for positional pairing",
                    inputs.len(),
                    outputs.len()
                );
            }
            for input in inputs {
                if !session.inputs().iter().any(|info| info.name == *input) {
                    anyhow::bail!(
                        "io.cross_kv_inputs declares decoder input '{input}' but the graph does not expose it; graph inputs: {:?}",
                        session.input_names()
                    );
                }
            }
            Ok(inputs
                .iter()
                .cloned()
                .zip(outputs.iter().cloned())
                .collect())
        }
        (None, None) => Ok(Vec::new()),
        _ => anyhow::bail!(
            "io.cross_kv_inputs and io.cross_kv_outputs must be declared together for positional pairing"
        ),
    }
}

fn resolve_position_program(
    session: &Session,
    io: &onnx_genai_metadata::ModelIoSpec,
    positions: Option<&PositionProgram>,
) -> anyhow::Result<Option<String>> {
    let Some(program) = positions else {
        return Ok(io.position_ids_input.clone());
    };
    if program.rank == 0 {
        anyhow::bail!("pipeline.positions.rank must be at least 1");
    }
    if let Some(io_input) = io.position_ids_input.as_deref()
        && io_input != program.input
    {
        anyhow::bail!(
            "pipeline.positions.input '{}' does not match decoder io.position_ids_input '{}'; declare the same graph port in both metadata sections",
            program.input,
            io_input
        );
    }
    if let Some(axes) = &program.axes
        && axes.len() != program.rank
    {
        anyhow::bail!(
            "pipeline.positions declares rank {} but {} axis labels {:?}; provide exactly one label per position axis",
            program.rank,
            axes.len(),
            axes
        );
    }
    if program
        .sections
        .as_ref()
        .is_some_and(|sections| sections.contains(&0))
    {
        anyhow::bail!("pipeline.positions.sections must contain only positive section sizes");
    }
    let dtype = program.dtype.as_deref().unwrap_or("int64");
    if dtype != "int64" {
        anyhow::bail!(
            "pipeline.positions declares dtype '{dtype}', but the engine currently supports generated position tensors only as int64"
        );
    }
    let continuation = program
        .continuation
        .as_deref()
        .unwrap_or("linear_increment");
    if !matches!(continuation, "linear_increment" | "carry_max" | "from_grid") {
        anyhow::bail!(
            "pipeline.positions declares unsupported continuation '{continuation}'; supported continuations: linear_increment, carry_max, from_grid"
        );
    }
    let input = session
        .inputs()
        .iter()
        .find(|info| info.name == program.input)
        .with_context(|| {
            format!(
                "pipeline.positions declares input '{}' but the decoder graph does not expose it; graph inputs: {:?}",
                program.input,
                session.input_names()
            )
        })?;
    ensure_i64(input)?;
    validate_position_shape(input, program.rank)?;
    Ok(Some(program.input.clone()))
}

pub(super) fn shapes_compatible(left: &[i64], right: &[i64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left <= &0 || right <= &0 || left == right)
}

pub(super) fn validate_position_shape(info: &TensorInfo, rank: usize) -> anyhow::Result<()> {
    let expected_tensor_rank = if rank == 1 { 2 } else { 3 };
    if info.shape.len() != expected_tensor_rank {
        anyhow::bail!(
            "position input '{}' has shape {:?}, but metadata rank {} requires tensor shape {}",
            info.name,
            info.shape,
            rank,
            if rank == 1 {
                "[batch, sequence]".to_string()
            } else {
                format!("[{rank}, batch, sequence]")
            }
        );
    }
    if rank > 1 && info.shape[0] > 0 && info.shape[0] != rank as i64 {
        anyhow::bail!(
            "position input '{}' has leading axis dimension {}, but pipeline.positions.rank is {}",
            info.name,
            info.shape[0],
            rank
        );
    }
    let batch_axis = usize::from(rank > 1);
    if info.shape[batch_axis] > 0 && info.shape[batch_axis] != 1 {
        anyhow::bail!(
            "position input '{}' has batch dimension {}, but the decode engine currently runs batch size 1",
            info.name,
            info.shape[batch_axis]
        );
    }
    Ok(())
}

impl ResolvedIo {
    /// Resolve port bindings from an explicit `io` block when present, else fall
    /// back to tensor-name conventions.
    pub(crate) fn resolve_with_positions(
        session: &Session,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
        positions: Option<&PositionProgram>,
    ) -> anyhow::Result<Self> {
        match io {
            Some(io) => Self::from_spec(session, io, positions),
            // TRANSITIONAL: remove in Phase 2 once all packages emit `io`.
            None => {
                if positions.is_some() {
                    anyhow::bail!(
                        "pipeline.positions requires an explicit decoder io block so its position input can be validated"
                    );
                }
                Ok(Self::default())
            }
        }
    }

    fn from_spec(
        session: &Session,
        io: &onnx_genai_metadata::ModelIoSpec,
        positions: Option<&PositionProgram>,
    ) -> anyhow::Result<Self> {
        let has_input = |name: &str| session.inputs().iter().any(|info| info.name == name);
        let has_output = |name: &str| session.outputs().iter().any(|info| info.name == name);

        for (label, port) in [
            ("io.token_input", &io.token_input),
            ("io.inputs_embeds_input", &io.inputs_embeds_input),
            ("io.attention_mask_input", &io.attention_mask_input),
            ("io.position_ids_input", &io.position_ids_input),
            (
                "io.encoder_hidden_states_input",
                &io.encoder_hidden_states_input,
            ),
        ] {
            if let Some(name) = port.as_deref().filter(|name| !has_input(name)) {
                anyhow::bail!(
                    "{label} declares input '{name}' but the graph does not expose it; graph inputs: {:?}",
                    session.input_names()
                );
            }
        }
        for (label, port) in [
            ("io.logits_output", &io.logits_output),
            ("io.hidden_output", &io.hidden_output),
        ] {
            if let Some(name) = port.as_deref().filter(|name| !has_output(name)) {
                anyhow::bail!(
                    "{label} declares output '{name}' but the graph does not expose it; graph outputs: {:?}",
                    session.output_names()
                );
            }
        }
        let kv_pairs = match (&io.kv_inputs, &io.kv_outputs) {
            (Some(inputs), Some(outputs)) => {
                if inputs.len() != outputs.len() {
                    anyhow::bail!(
                        "io.kv_inputs ({}) and io.kv_outputs ({}) must have equal length for positional pairing",
                        inputs.len(),
                        outputs.len()
                    );
                }
                for name in inputs {
                    if !has_input(name) {
                        anyhow::bail!(
                            "io.kv_inputs declares input '{name}' but the graph does not expose it; graph inputs: {:?}",
                            session.input_names()
                        );
                    }
                }
                for name in outputs {
                    if !has_output(name) {
                        anyhow::bail!(
                            "io.kv_outputs declares output '{name}' but the graph does not expose it; graph outputs: {:?}",
                            session.output_names()
                        );
                    }
                }
                inputs
                    .iter()
                    .cloned()
                    .zip(outputs.iter().cloned())
                    .collect()
            }
            (None, None) => Vec::new(),
            _ => anyhow::bail!(
                "io.kv_inputs and io.kv_outputs must be declared together (positional KV pairing)"
            ),
        };
        if let Some(update) = io.kv_update.as_deref()
            && !matches!(update, "append" | "shared_buffer")
        {
            anyhow::bail!(
                "io.kv_update declares unsupported update '{update}'; supported KV updates: append, shared_buffer"
            );
        }
        let cross_kv_pairs = resolve_cross_kv_pairs(
            session,
            io.cross_kv_inputs.as_deref(),
            io.cross_kv_outputs.as_deref(),
        )?;

        let state_pairs = resolve_state_pairs(session, io.state_pairs.as_deref(), &kv_pairs)?;
        let position_ids_input = resolve_position_program(session, io, positions)?;

        Ok(Self {
            explicit: true,
            token_input: io.token_input.clone(),
            inputs_embeds_input: io.inputs_embeds_input.clone(),
            attention_mask_input: io.attention_mask_input.clone(),
            position_ids_input,
            logits_output: io.logits_output.clone(),
            hidden_output: io.hidden_output.clone(),
            kv_pairs,
            state_pairs,
            cross_kv_pairs,
        })
    }

    /// Whether `name` is the token-id input for this graph.
    pub(super) fn is_token_input(&self, name: &str, lower: &str) -> bool {
        if self.explicit {
            self.token_input.as_deref() == Some(name)
        } else {
            // TRANSITIONAL: remove in Phase 2 once all packages emit `io`.
            is_token_input_name(lower)
        }
    }

    /// Whether `name` is the attention-mask input for this graph.
    pub(super) fn is_attention_mask_input(&self, name: &str, lower: &str) -> bool {
        if self.explicit {
            self.attention_mask_input.as_deref() == Some(name)
        } else {
            // TRANSITIONAL: remove in Phase 2 once all packages emit `io`.
            lower == "attention_mask" || lower.ends_with(".attention_mask")
        }
    }

    /// Whether `name` is the position-ids input for this graph.
    pub(super) fn is_position_ids_input(&self, name: &str, lower: &str) -> bool {
        if self.explicit {
            self.position_ids_input.as_deref() == Some(name)
        } else {
            // TRANSITIONAL: remove in Phase 2 once all packages emit `io`.
            lower == "position_ids" || lower.ends_with(".position_ids")
        }
    }
}
