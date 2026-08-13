//! Resolved decode-step graph I/O port bindings.
//!
//! Pure code motion from `decode.rs`: I/O resolution for the decode step.

use super::values::ensure_i64;
use super::*;
use onnx_genai_ort::io_roles::{
    is_rank_one_or_two_sequence, is_rank_one_to_three_output, is_rank_three_sequence, resolve_port,
};

/// Resolved graph I/O port bindings for the decode step.
///
/// Built from explicit metadata and otherwise from unambiguous tensor shapes.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResolvedIo {
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
}

fn resolve_state_pairs(
    session: &dyn GraphIo,
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
        if input.shape.is_empty() {
            anyhow::bail!(
                "state input '{}' has scalar shape; loop-carried state requires at least a batch axis",
                pair.input,
            );
        }
        if input
            .shape
            .iter()
            .enumerate()
            .any(|(axis, dimension)| *dimension <= 0 && axis != 0)
        {
            anyhow::bail!(
                "state input '{}' has dynamic or invalid non-batch shape {:?}; zero initialization requires every non-batch fixed-state dimension to be concrete and positive (the leading batch axis may be symbolic and resolves to the decode batch)",
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
fn resolve_position_program(
    session: &dyn GraphIo,
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
    /// Resolve port bindings from explicit metadata or unambiguous tensor shape.
    pub(crate) fn resolve_with_positions(
        session: &dyn GraphIo,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
        positions: Option<&PositionProgram>,
    ) -> anyhow::Result<Self> {
        match io {
            Some(io) => Self::from_spec(session, io, positions),
            None => {
                if positions.is_some() {
                    anyhow::bail!(
                        "pipeline.positions requires an explicit decoder io block so its position input can be validated"
                    );
                }
                Self::from_structure(session)
            }
        }
    }

    fn from_structure(session: &dyn GraphIo) -> anyhow::Result<Self> {
        let input =
            |role: &str, structural: fn(&TensorInfo) -> bool| -> anyhow::Result<Option<String>> {
                resolve_port(session.inputs(), None, role, structural)
                    .map_err(anyhow::Error::msg)
                    .map(|resolved| resolved.map(|resolved| resolved.name))
            };
        let output =
            |role: &str, structural: fn(&TensorInfo) -> bool| -> anyhow::Result<Option<String>> {
                resolve_port(session.outputs(), None, role, structural)
                    .map_err(anyhow::Error::msg)
                    .map(|resolved| resolved.map(|resolved| resolved.name))
            };
        let token_input = input("model.io.token_input", is_rank_one_or_two_sequence)?;
        let inputs_embeds_input = input("model.io.inputs_embeds_input", is_rank_three_sequence)?;
        if token_input.is_none() && inputs_embeds_input.is_none() {
            anyhow::bail!(
                "cannot resolve the decoder sequence input from tensor shape; declare model.io.sequence_source and its exact model.io.token_input or model.io.inputs_embeds_input"
            );
        }
        let logits_output =
            output("model.io.logits_output", is_rank_one_to_three_output)?.with_context(|| {
                "cannot resolve decoder logits from tensor shape; declare model.io.logits_output"
            })?;
        if session.inputs().iter().any(|info| {
            matches!(
                info.dtype,
                DataType::Float16 | DataType::BFloat16 | DataType::Float32
            ) && info.shape.len() >= 3
                && Some(info.name.as_str()) != inputs_embeds_input.as_deref()
        }) {
            anyhow::bail!(
                "decoder exposes stateful floating-point inputs that cannot be paired unambiguously by shape; declare model.io.kv_inputs and model.io.kv_outputs (or model.io.state_pairs)"
            );
        }
        Ok(Self {
            token_input,
            inputs_embeds_input,
            attention_mask_input: None,
            position_ids_input: None,
            logits_output: Some(logits_output),
            hidden_output: None,
            kv_pairs: Vec::new(),
            state_pairs: Vec::new(),
        })
    }

    fn from_spec(
        session: &dyn GraphIo,
        io: &onnx_genai_metadata::ModelIoSpec,
        positions: Option<&PositionProgram>,
    ) -> anyhow::Result<Self> {
        let has_input = |name: &str| session.inputs().iter().any(|info| info.name == name);
        let has_output = |name: &str| session.outputs().iter().any(|info| info.name == name);
        let occupied_inputs = [
            io.attention_mask_input.as_deref(),
            io.position_ids_input.as_deref(),
            io.encoder_hidden_states_input.as_deref(),
            io.audio_features_input.as_deref(),
        ]
        .into_iter()
        .flatten()
        .chain(io.kv_inputs.iter().flatten().map(String::as_str))
        .chain(io.cross_kv_inputs.iter().flatten().map(String::as_str))
        .chain(
            io.state_pairs
                .iter()
                .flatten()
                .map(|pair| pair.input.as_str()),
        )
        .chain(io.optional_inputs.keys().map(String::as_str))
        .collect::<HashSet<_>>();
        let resolve_input =
            |declared: Option<&str>, key: &str, structural: fn(&TensorInfo) -> bool| {
                resolve_port(session.inputs(), declared, key, |tensor| {
                    !occupied_inputs.contains(tensor.name.as_str()) && structural(tensor)
                })
                .map_err(anyhow::Error::msg)
                .map(|port| port.map(|port| port.name))
            };
        let sequence_source = io
            .sequence_source
            .unwrap_or(onnx_genai_metadata::SequenceInputKind::TokenIds);
        let (token_input, inputs_embeds_input) = match sequence_source {
            onnx_genai_metadata::SequenceInputKind::TokenIds => (
                Some(
                    resolve_input(
                        io.token_input.as_deref(),
                        "model.io.token_input",
                        is_rank_one_or_two_sequence,
                    )?
                    .context(
                        "cannot resolve decoder token input from tensor shape; declare model.io.token_input",
                    )?,
                ),
                io.inputs_embeds_input.clone(),
            ),
            onnx_genai_metadata::SequenceInputKind::InputsEmbeds => (
                io.token_input.clone(),
                Some(
                    resolve_input(
                        io.inputs_embeds_input.as_deref(),
                        "model.io.inputs_embeds_input",
                        is_rank_three_sequence,
                    )?
                    .context(
                        "cannot resolve decoder embedding input from tensor shape; declare model.io.inputs_embeds_input",
                    )?,
                ),
            ),
        };
        let occupied_outputs = io
            .hidden_output
            .iter()
            .map(String::as_str)
            .chain(io.kv_outputs.iter().flatten().map(String::as_str))
            .chain(
                io.state_pairs
                    .iter()
                    .flatten()
                    .map(|pair| pair.output.as_str()),
            )
            .collect::<HashSet<_>>();
        let logits_output = resolve_port(
            session.outputs(),
            io.logits_output.as_deref(),
            "model.io.logits_output",
            |tensor| {
                !occupied_outputs.contains(tensor.name.as_str())
                    && is_rank_one_to_three_output(tensor)
            },
        )
        .map_err(anyhow::Error::msg)?
        .map(|port| port.name)
        .context(
            "cannot resolve decoder logits from tensor shape; declare model.io.logits_output",
        )?;

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
        let state_pairs = resolve_state_pairs(session, io.state_pairs.as_deref(), &kv_pairs)?;
        let position_ids_input = resolve_position_program(session, io, positions)?;

        Ok(Self {
            token_input,
            inputs_embeds_input,
            attention_mask_input: io.attention_mask_input.clone(),
            position_ids_input,
            logits_output: Some(logits_output),
            hidden_output: io.hidden_output.clone(),
            kv_pairs,
            state_pairs,
        })
    }

    /// Whether `name` is the token-id input for this graph.
    pub(super) fn is_token_input(&self, name: &str) -> bool {
        self.token_input.as_deref() == Some(name)
    }

    /// Whether `name` is the attention-mask input for this graph.
    pub(super) fn is_attention_mask_input(&self, name: &str) -> bool {
        self.attention_mask_input.as_deref() == Some(name)
    }

    /// Whether `name` is the position-ids input for this graph.
    pub(super) fn is_position_ids_input(&self, name: &str) -> bool {
        self.position_ids_input.as_deref() == Some(name)
    }
}
