//! Universal typed workflow interpreter.

use super::*;
use crate::decode::clone_value;

type WorkflowAdapterExecutor = fn(
    &PipelineEngine,
    &str,
    &std::collections::BTreeMap<String, String>,
    &std::collections::BTreeMap<String, String>,
    &onnx_genai_metadata::WorkflowComponent,
    &mut PipelineTensors,
    &HashMap<String, i64>,
    &mut HashMap<String, i64>,
) -> anyhow::Result<()>;

fn workflow_adapter_registry()
-> &'static HashMap<(&'static str, &'static str), WorkflowAdapterExecutor> {
    static REGISTRY: std::sync::LazyLock<
        HashMap<(&'static str, &'static str), WorkflowAdapterExecutor>,
    > = std::sync::LazyLock::new(|| {
        HashMap::from([(
            ("onnx-genai.image-preprocess", "1"),
            PipelineEngine::run_image_preprocess_adapter as WorkflowAdapterExecutor,
        )])
    });
    &REGISTRY
}

pub(super) fn supports_workflow_adapter(abi: &str, version: &str) -> bool {
    workflow_adapter_registry().contains_key(&(abi, version))
}

impl PipelineEngine {
    pub(crate) fn run_workflow(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelineGenerateRequest {
            request,
            inputs,
            session_id,
        } = request;
        let workflow = &self.workflow;
        let mut values = self.bind_workflow_inputs(workflow, &request, inputs)?;
        for (cell, state) in &workflow.state {
            if state.scope != onnx_genai_metadata::WorkflowStateScope::Session {
                continue;
            }
            let session_id = session_id.as_ref().with_context(|| {
                format!("session-scoped workflow state '{cell}' requires a session id")
            })?;
            if let Some(value) = self
                .workflow_session_state
                .borrow()
                .get(&(session_id.clone(), cell.clone()))
            {
                values.insert(state.initializer.clone(), clone_value(value)?);
            }
        }
        let mut symbols = HashMap::new();
        let dynamic_symbols = workflow
            .state
            .values()
            .filter_map(|state| match &state.recurrence {
                onnx_genai_metadata::ShapeRecurrence::Growing { axis, .. }
                | onnx_genai_metadata::ShapeRecurrence::Bounded { axis, .. } => state
                    .contract
                    .shape
                    .as_ref()
                    .and_then(|shape| shape.get(*axis))
                    .and_then(|dimension| match dimension {
                        TensorDimension::Symbol(symbol) => Some(symbol.clone()),
                        TensorDimension::Fixed(_) => None,
                    }),
                onnx_genai_metadata::ShapeRecurrence::Invariant => None,
            })
            .collect::<std::collections::HashSet<_>>();
        for (name, input) in &workflow.inputs {
            if let Some(value) = values.get(name) {
                validate_workflow_value(
                    name,
                    value,
                    &input.contract,
                    &mut symbols,
                    &dynamic_symbols,
                )?;
            }
        }
        let mut emit_counts = HashMap::new();
        let mut final_state_refs = HashMap::new();
        self.run_workflow_node(
            &workflow.graph,
            workflow,
            &mut values,
            &mut symbols,
            &dynamic_symbols,
            &mut emit_counts,
            &mut final_state_refs,
        )?;
        for output in workflow_emitted_outputs(&workflow.graph) {
            let Some(value) = values.get(&output) else {
                continue;
            };
            let contract = &workflow
                .outputs
                .get(&output)
                .with_context(|| format!("workflow emitted undeclared output '{output}'"))?
                .contract;
            validate_workflow_value(&output, value, contract, &mut symbols, &dynamic_symbols)?;
        }
        if let Some(session_id) = session_id {
            let mut updates = Vec::new();
            for (cell, state) in &workflow.state {
                if state.scope != onnx_genai_metadata::WorkflowStateScope::Session {
                    continue;
                }
                let value_ref = final_state_refs
                    .get(cell)
                    .map(String::as_str)
                    .unwrap_or(&state.initializer);
                let value = values.get(value_ref).with_context(|| {
                    format!(
                        "session-scoped workflow state '{cell}' has no final value '{value_ref}'"
                    )
                })?;
                updates.push(((session_id.clone(), cell.clone()), clone_value(value)?));
            }
            let mut session_state = self.workflow_session_state.borrow_mut();
            for (key, value) in updates {
                session_state.insert(key, value);
            }
        }
        Ok(values)
    }
    fn run_workflow_node(
        &self,
        node: &WorkflowNode,
        workflow: &WorkflowSpec,
        values: &mut PipelineTensors,
        symbols: &mut HashMap<String, i64>,
        dynamic_symbols: &std::collections::HashSet<String>,
        emit_counts: &mut HashMap<String, usize>,
        final_state_refs: &mut HashMap<String, String>,
    ) -> anyhow::Result<()> {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    self.run_workflow_node(
                        node,
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                    )?;
                }
            }
            WorkflowNode::Invoke {
                component,
                inputs,
                outputs,
                ..
            } => {
                let declaration = workflow
                    .components
                    .get(component)
                    .with_context(|| format!("workflow component '{component}' is undeclared"))?;
                match &declaration.implementation {
                    ComponentImplementation::Onnx { .. } => {
                        let session = self.models.session(component).with_context(|| {
                            format!("workflow ONNX component '{component}' was not loaded")
                        })?;
                        // Component dimensions are invocation-local. A decoder, for example, may
                        // bind `sequence` to the prompt length in setup and to one in the loop.
                        // Values crossing the package boundary were already checked there; the
                        // component contract unifies its own ports without conflating equal
                        // spelling in separate contract scopes.
                        let mut component_symbols = HashMap::new();
                        let component_dynamic_symbols = std::collections::HashSet::new();
                        let resolved = inputs
                            .iter()
                            .map(|(port, value)| {
                                values
                                    .get(value)
                                    .with_context(|| {
                                        format!(
                                            "workflow component '{component}' input '{port}' \
                                             references unavailable value '{value}'"
                                        )
                                    })
                                    .and_then(|tensor| {
                                        let contract =
                                            declaration.ports.inputs.get(port).with_context(
                                                || {
                                                    format!(
                                                        "workflow component '{component}' has no \
                                                     declared input port '{port}'"
                                                    )
                                                },
                                            )?;
                                        validate_workflow_value(
                                            value,
                                            tensor,
                                            contract,
                                            &mut component_symbols,
                                            &component_dynamic_symbols,
                                        )?;
                                        Ok((port.as_str(), tensor))
                                    })
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?;
                        let produced = session.run(&resolved)?;
                        for (port, tensor) in session.output_names().iter().zip(produced) {
                            let value = outputs.get(port).with_context(|| {
                                format!(
                                    "workflow component '{component}' output '{port}' has no SSA binding"
                                )
                            })?;
                            let contract =
                                declaration.ports.outputs.get(port).with_context(|| {
                                    format!(
                                        "workflow component '{component}' has no declared output \
                                         port '{port}'"
                                    )
                                })?;
                            validate_workflow_value(
                                value,
                                &tensor,
                                contract,
                                &mut component_symbols,
                                &component_dynamic_symbols,
                            )?;
                            values.insert(value.clone(), tensor);
                        }
                    }
                    ComponentImplementation::Binding => {
                        for (port, output) in outputs {
                            let source = inputs.get(port).with_context(|| {
                                format!(
                                    "binding component '{component}' output '{port}' requires \
                                     an input with the same port name"
                                )
                            })?;
                            let tensor = values.get(source).with_context(|| {
                                format!("binding source value '{source}' is unavailable")
                            })?;
                            values.insert(output.clone(), clone_value(tensor)?);
                        }
                    }
                    ComponentImplementation::Adapter { abi, version, .. } => {
                        if let Some(execute) =
                            workflow_adapter_registry().get(&(abi.as_str(), version.as_str()))
                        {
                            let mut component_symbols = HashMap::new();
                            execute(
                                self,
                                component,
                                inputs,
                                outputs,
                                declaration,
                                values,
                                symbols,
                                &mut component_symbols,
                            )?;
                        } else {
                            anyhow::bail!(
                                "workflow adapter '{component}' requires unsupported ABI \
                                 {abi}@{version}"
                            );
                        }
                    }
                }
            }
            WorkflowNode::Loop {
                setup,
                body,
                condition,
                max_iterations,
                iteration,
                carried,
            } => {
                self.run_workflow_node(
                    setup,
                    workflow,
                    values,
                    symbols,
                    dynamic_symbols,
                    emit_counts,
                    final_state_refs,
                )?;
                for carry in carried {
                    let state = workflow.state.get(&carry.cell).with_context(|| {
                        format!("workflow loop carries undeclared state '{}'", carry.cell)
                    })?;
                    let initializer = values.get(&state.initializer).with_context(|| {
                        format!(
                            "workflow state '{}' initializer '{}' is unavailable after loop setup",
                            carry.cell, state.initializer
                        )
                    })?;
                    validate_workflow_value(
                        &state.initializer,
                        initializer,
                        &state.contract,
                        symbols,
                        dynamic_symbols,
                    )?;
                    final_state_refs.insert(carry.cell.clone(), carry.current.clone());
                }
                let limit = workflow_scalar_usize(values, max_iterations)?;
                if let Some(iteration) = iteration
                    && values.contains_key(&iteration.value)
                {
                    anyhow::bail!(
                        "workflow loop iteration value '{}' shadows an existing SSA value",
                        iteration.value
                    );
                }
                for index in 0..limit {
                    if let Some(iteration) = iteration {
                        values.insert(
                            iteration.value.clone(),
                            workflow_iteration_value(index, &iteration.contract, symbols)?,
                        );
                    }
                    for carry in carried {
                        let current = values.get(&carry.current).with_context(|| {
                            format!("workflow loop value '{}' is unavailable", carry.current)
                        })?;
                        values.insert(carry.body_input.clone(), clone_value(current)?);
                    }
                    self.run_workflow_node(
                        body,
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                    )?;
                    for carry in carried {
                        let current = values.get(&carry.current).with_context(|| {
                            format!("workflow loop value '{}' is unavailable", carry.current)
                        })?;
                        let next = values.get(&carry.body_output).with_context(|| {
                            format!("workflow loop body did not produce '{}'", carry.body_output)
                        })?;
                        let state = workflow.state.get(&carry.cell).with_context(|| {
                            format!("workflow loop carries undeclared state '{}'", carry.cell)
                        })?;
                        validate_state_recurrence(&carry.cell, current, next, state, values)?;
                        let next_value = clone_value(next)?;
                        values.insert(carry.current.clone(), clone_value(&next_value)?);
                        values.insert(carry.next.clone(), next_value);
                        final_state_refs.insert(carry.cell.clone(), carry.next.clone());
                    }
                    if !workflow_scalar_bool(values, condition)? {
                        break;
                    }
                }
                if let Some(iteration) = iteration {
                    values.remove(&iteration.value);
                }
            }
            WorkflowNode::Branch {
                predicate,
                cases,
                default,
                outputs,
                effects: _,
            } => {
                let key = workflow_scalar_key(values, predicate)?;
                let (selected, is_default) = if let Some(case) = cases.get(&key) {
                    (case, false)
                } else if let Some(default) = default {
                    (default.as_ref(), true)
                } else {
                    anyhow::bail!("workflow branch has no case '{key}' and no default");
                };
                let mut branch_values = clone_pipeline_tensors(values)?;
                let mut branch_state_refs = final_state_refs.clone();
                let emit_counts_before = emit_counts.clone();
                self.run_workflow_node(
                    selected,
                    workflow,
                    &mut branch_values,
                    symbols,
                    dynamic_symbols,
                    emit_counts,
                    &mut branch_state_refs,
                )?;

                // Emits are explicit side effects at the package boundary, so selected-branch
                // output values and event records survive even though ordinary case SSA does not.
                for output in workflow_emitted_outputs(selected) {
                    if let Some(value) = branch_values.get(&output) {
                        values.insert(output.clone(), clone_value(value)?);
                    }
                    let start = emit_counts_before.get(&output).copied().unwrap_or_default();
                    let end = emit_counts.get(&output).copied().unwrap_or_default();
                    for index in start..end {
                        let event = format!("{output}.{index}");
                        if let Some(value) = branch_values.get(&event) {
                            values.insert(event, clone_value(value)?);
                        }
                    }
                }

                for (output, phi) in outputs {
                    let source = if is_default {
                        phi.default.as_ref().with_context(|| {
                            format!("workflow branch output '{output}' has no default value")
                        })?
                    } else {
                        phi.cases.get(&key).with_context(|| {
                            format!(
                                "workflow branch output '{output}' has no value for case '{key}'"
                            )
                        })?
                    };
                    let value = branch_values.get(source).with_context(|| {
                        format!(
                            "workflow branch output '{output}' selected unavailable value \
                             '{source}'"
                        )
                    })?;
                    values.insert(output.clone(), clone_value(value)?);
                    for (cell, state_ref) in &branch_state_refs {
                        if state_ref == source {
                            final_state_refs.insert(cell.clone(), output.clone());
                        }
                    }
                }
            }
            WorkflowNode::Emit {
                value,
                valid_length,
                output,
                mode,
                ..
            } => {
                let tensor = values
                    .get(value)
                    .with_context(|| format!("workflow emit value '{value}' is unavailable"))?;
                let output_contract = workflow.outputs.get(output).with_context(|| {
                    format!("workflow emit references undeclared output '{output}'")
                })?;
                let emitted = if let Some(valid_length) = valid_length {
                    let length =
                        workflow_scalar_usize(values, valid_length).with_context(|| {
                            format!("workflow emit valid_length '{valid_length}' is invalid")
                        })?;
                    slice_workflow_prefix(tensor, length)?
                } else {
                    clone_value(tensor)?
                };
                let validation_contract =
                    if valid_length.is_some() || matches!(mode, WorkflowEmitMode::Append) {
                        emit_chunk_contract(&output_contract.contract, &emitted)?
                    } else {
                        output_contract.contract.clone()
                    };
                validate_workflow_value(
                    value,
                    &emitted,
                    &validation_contract,
                    symbols,
                    dynamic_symbols,
                )?;
                match mode {
                    WorkflowEmitMode::Replace => {
                        values.insert(output.clone(), emitted);
                    }
                    WorkflowEmitMode::Append => {
                        let appended = if let Some(previous) = values.get(output) {
                            append_workflow_value(previous, &emitted)?
                        } else {
                            emitted
                        };
                        values.insert(output.clone(), appended);
                    }
                    WorkflowEmitMode::Event => {
                        let index = emit_counts.entry(output.clone()).or_default();
                        values.insert(format!("{output}.{index}"), clone_value(&emitted)?);
                        *index += 1;
                        values.insert(output.clone(), emitted);
                    }
                }
            }
            WorkflowNode::Transfer {
                input,
                output,
                device,
            } => {
                if *device != DeviceKind::Cpu {
                    anyhow::bail!(
                        "workflow transfer to {device:?} requires a device allocator contract"
                    );
                }
                let tensor = values
                    .get(input)
                    .with_context(|| format!("workflow transfer value '{input}' is unavailable"))?;
                values.insert(output.clone(), clone_value(tensor)?);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_image_preprocess_adapter(
        &self,
        component: &str,
        inputs: &std::collections::BTreeMap<String, String>,
        outputs: &std::collections::BTreeMap<String, String>,
        declaration: &onnx_genai_metadata::WorkflowComponent,
        values: &mut PipelineTensors,
        package_symbols: &HashMap<String, i64>,
        component_symbols: &mut HashMap<String, i64>,
    ) -> anyhow::Result<()> {
        let program = self
            .preprocessing
            .as_ref()
            .and_then(|spec| spec.image.as_ref())
            .with_context(|| {
                format!(
                    "workflow image preprocessing adapter '{component}' requires \
                     preprocessing.image metadata"
                )
            })?;
        let encoded_ref = inputs.get("encoded").with_context(|| {
            format!("workflow image preprocessing adapter '{component}' requires input 'encoded'")
        })?;
        let encoded = values
            .get(encoded_ref)
            .with_context(|| format!("workflow value '{encoded_ref}' is unavailable"))?;
        let encoded_contract = declaration.ports.inputs.get("encoded").with_context(|| {
            format!(
                "workflow image preprocessing adapter '{component}' has no declared input \
                 port 'encoded'"
            )
        })?;
        let component_dynamic_symbols = std::collections::HashSet::new();
        validate_workflow_value(
            encoded_ref,
            encoded,
            encoded_contract,
            component_symbols,
            &component_dynamic_symbols,
        )?;
        if encoded.dtype() != DataType::Uint8 || encoded.shape().len() != 1 {
            anyhow::bail!(
                "workflow image preprocessing adapter '{component}' input 'encoded' must be \
                 uint8 rank 1"
            );
        }
        let pixels = program
            .outputs
            .iter()
            .find(|output| output.content == "pixels")
            .context("preprocessing.image must declare a pixels output")?;
        let pixel_contract = pixels.contract.as_ref().with_context(|| {
            format!(
                "preprocessing.image output '{}' must declare a TensorContract",
                pixels.name
            )
        })?;
        let target_shape = resolve_workflow_shape(pixel_contract, package_symbols)?;
        let processor = onnx_genai_preprocess::image::ImagePreprocessor::from_input_and_program(
            &target_shape,
            program,
        )?;
        let encoded_bytes = encoded.as_raw_bytes()?;
        let mut tensors = processor
            .preprocess_encoded([encoded_bytes])?
            .tensors
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<HashMap<_, _>>();
        for (port, value_ref) in outputs {
            let tensor = tensors.remove(value_ref).with_context(|| {
                format!(
                    "image preprocessing adapter '{component}' did not produce declared SSA \
                     output '{value_ref}'"
                )
            })?;
            let contract = declaration.ports.outputs.get(port).with_context(|| {
                format!(
                    "workflow image preprocessing adapter '{component}' has no declared output \
                     port '{port}'"
                )
            })?;
            let value = image_tensor_to_value(tensor)?;
            validate_workflow_value(
                value_ref,
                &value,
                contract,
                component_symbols,
                &component_dynamic_symbols,
            )?;
            values.insert(value_ref.clone(), value);
        }
        if !tensors.is_empty() {
            let mut unbound = tensors.into_keys().collect::<Vec<_>>();
            unbound.sort();
            anyhow::bail!(
                "image preprocessing adapter '{component}' produced outputs without workflow \
                 bindings: {}",
                unbound.join(", ")
            );
        }
        Ok(())
    }

    fn bind_workflow_inputs(
        &self,
        workflow: &WorkflowSpec,
        request: &GenerateRequest,
        mut provided: PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        let mut values = HashMap::new();
        for (name, input) in &workflow.inputs {
            let supplied = provided.remove(name).or_else(|| match &input.source {
                WorkflowInputSource::Application { name } => provided.remove(name),
                _ => None,
            });
            let value = if let Some(value) = supplied {
                Some(value)
            } else {
                match &input.source {
                    WorkflowInputSource::Request { field } => {
                        workflow_request_value(field, request, &input.contract)?
                    }
                    WorkflowInputSource::Literal => input
                        .default
                        .as_ref()
                        .map(|value| workflow_literal_value(value, &input.contract))
                        .transpose()?,
                    WorkflowInputSource::Application { .. } => input
                        .default
                        .as_ref()
                        .map(|value| workflow_literal_value(value, &input.contract))
                        .transpose()?,
                    WorkflowInputSource::Artifact { path } => {
                        anyhow::bail!(
                            "workflow input '{name}' requires artifact binding '{path}', which \
                             is not a tensor request input"
                        )
                    }
                }
            };
            match value {
                Some(value) => {
                    values.insert(name.clone(), value);
                }
                None if input.required => {
                    anyhow::bail!("required workflow package input '{name}' was not supplied")
                }
                None => {}
            }
        }
        if !provided.is_empty() {
            anyhow::bail!(
                "workflow request supplied undeclared application inputs: {:?}",
                provided.keys().collect::<Vec<_>>()
            );
        }
        Ok(values)
    }
}

fn workflow_request_value(
    field: &RuntimeInputRole,
    request: &GenerateRequest,
    contract: &TensorContract,
) -> anyhow::Result<Option<Value>> {
    let scalar_i64 = |value: i64| {
        let shape = scalar_or_batch_shape(contract)?;
        Value::from_slice_i64(&vec![value; shape_numel(&shape)], &shape).map_err(Into::into)
    };
    let scalar_f32 = |value: f32| {
        let shape = scalar_or_batch_shape(contract)?;
        Value::from_slice_f32(&vec![value; shape_numel(&shape)], &shape).map_err(Into::into)
    };
    match field {
        RuntimeInputRole::PromptTokens => match &request.prompt {
            GeneratePrompt::TokenIds(tokens) => {
                let data = tokens
                    .iter()
                    .map(|token| i64::from(*token))
                    .collect::<Vec<_>>();
                let shape = match contract.rank {
                    1 => vec![data.len() as i64],
                    2 => vec![1, data.len() as i64],
                    rank => anyhow::bail!(
                        "prompt token workflow input must have rank 1 or 2, got {rank}"
                    ),
                };
                Ok(Some(Value::from_slice_i64(&data, &shape)?))
            }
            GeneratePrompt::Text(_) => anyhow::bail!(
                "prompt_tokens request binding requires token ids; use a tokenizer adapter for text"
            ),
        },
        RuntimeInputRole::PromptText => {
            anyhow::bail!("prompt_text request binding requires a versioned tokenizer adapter")
        }
        RuntimeInputRole::MaxIterations | RuntimeInputRole::MaxOutputTokens => {
            scalar_i64(request.options.max_new_tokens as i64).map(Some)
        }
        RuntimeInputRole::Seed => {
            scalar_i64(request.options.seed.unwrap_or_default() as i64).map(Some)
        }
        RuntimeInputRole::SamplingTemperature => scalar_f32(request.options.temperature).map(Some),
        RuntimeInputRole::SamplingTopK => scalar_i64(request.options.top_k as i64).map(Some),
        RuntimeInputRole::SamplingTopP => scalar_f32(request.options.top_p).map(Some),
        RuntimeInputRole::Media | RuntimeInputRole::Constraint | RuntimeInputRole::SessionId => {
            Ok(None)
        }
    }
}

fn workflow_literal_value(
    scalar: &ScalarValue,
    contract: &TensorContract,
) -> anyhow::Result<Value> {
    let shape = literal_shape(contract)?;
    let numel = shape_numel(&shape);
    match scalar {
        ScalarValue::Integer(value) => {
            let (bytes, dtype) = match contract.dtype.as_str() {
                "int64" => (value.to_le_bytes().repeat(numel), DataType::Int64),
                "int32" => (
                    i32::try_from(*value)
                        .context("integer literal exceeds int32")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Int32,
                ),
                "int16" => (
                    i16::try_from(*value)
                        .context("integer literal exceeds int16")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Int16,
                ),
                "int8" => (
                    vec![
                        i8::try_from(*value).context("integer literal exceeds int8")? as u8;
                        numel
                    ],
                    DataType::Int8,
                ),
                "uint64" => (
                    u64::try_from(*value)
                        .context("integer literal is negative")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Uint64,
                ),
                "uint32" => (
                    u32::try_from(*value)
                        .context("integer literal exceeds uint32")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Uint32,
                ),
                "uint16" => (
                    u16::try_from(*value)
                        .context("integer literal exceeds uint16")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Uint16,
                ),
                "uint8" => (
                    vec![u8::try_from(*value).context("integer literal exceeds uint8")?; numel],
                    DataType::Uint8,
                ),
                _ => anyhow::bail!(
                    "integer workflow literal is incompatible with declared dtype '{}'",
                    contract.dtype
                ),
            };
            Value::from_raw_bytes(bytes, &shape, dtype).map_err(Into::into)
        }
        ScalarValue::Float(value) => {
            let (bytes, dtype) = match contract.dtype.as_str() {
                "float32" | "fp32" => (
                    (*value as f32).to_le_bytes().repeat(numel),
                    DataType::Float32,
                ),
                "float16" | "fp16" => (
                    half::f16::from_f64(*value)
                        .to_bits()
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Float16,
                ),
                "bfloat16" | "bf16" => (
                    half::bf16::from_f64(*value)
                        .to_bits()
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::BFloat16,
                ),
                _ => anyhow::bail!(
                    "floating-point workflow literal is incompatible with declared dtype '{}'",
                    contract.dtype
                ),
            };
            Value::from_raw_bytes(bytes, &shape, dtype).map_err(Into::into)
        }
        ScalarValue::Bool(value) if contract.dtype == "bool" => {
            Value::from_raw_bytes(vec![u8::from(*value); numel], &shape, DataType::Bool)
                .map_err(Into::into)
        }
        ScalarValue::String(_) => {
            anyhow::bail!("string literal workflow inputs require an adapter binding")
        }
        _ => anyhow::bail!(
            "workflow literal is incompatible with declared dtype '{}'",
            contract.dtype
        ),
    }
}

fn scalar_or_batch_shape(contract: &TensorContract) -> anyhow::Result<Vec<i64>> {
    match contract.rank {
        0 => Ok(Vec::new()),
        1 => Ok(vec![1]),
        rank => anyhow::bail!("request scalar binding requires rank 0 or 1, got {rank}"),
    }
}

fn resolve_workflow_shape(
    contract: &TensorContract,
    symbols: &HashMap<String, i64>,
) -> anyhow::Result<Vec<i64>> {
    let shape = contract
        .shape
        .as_ref()
        .context("workflow adapter output requires a declared shape")?;
    shape
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Fixed(value) => Ok(*value),
            TensorDimension::Symbol(symbol) => symbols.get(symbol).copied().with_context(|| {
                format!(
                    "workflow adapter output requires unresolved dimension '{symbol}' for allocation"
                )
            }),
        })
        .collect()
}

fn workflow_iteration_value(
    index: usize,
    contract: &TensorContract,
    symbols: &HashMap<String, i64>,
) -> anyhow::Result<Value> {
    if contract.dtype != "int64" {
        anyhow::bail!(
            "workflow loop iteration requires dtype int64, got '{}'",
            contract.dtype
        );
    }
    let index = i64::try_from(index).context("workflow loop iteration exceeds int64")?;
    match contract.rank {
        0 => Value::from_slice_i64(&[index], &[]).map_err(Into::into),
        1 => {
            let shape = resolve_workflow_shape(contract, symbols)?;
            let elements = shape_numel(&shape);
            Value::from_vec_i64(vec![index; elements], &shape).map_err(Into::into)
        }
        rank => anyhow::bail!("workflow loop iteration requires rank 0 or rank 1, got rank {rank}"),
    }
}

fn image_tensor_to_value(
    tensor: onnx_genai_preprocess::image::NamedImageTensor,
) -> anyhow::Result<Value> {
    use onnx_genai_preprocess::image::ImageTensorData;

    let shape = &tensor.shape;
    match tensor.data {
        ImageTensorData::Fp32(data) => Value::from_vec_f32(data, shape).map_err(Into::into),
        ImageTensorData::Fp16(data) => Value::from_vec_f16_bits(data, shape).map_err(Into::into),
        ImageTensorData::Bf16(data) => Value::from_vec_bf16_bits(data, shape).map_err(Into::into),
        ImageTensorData::Int64(data) => Value::from_vec_i64(data, shape).map_err(Into::into),
        ImageTensorData::Int32(data) => {
            typed_image_bytes(data, shape, DataType::Int32, i32::to_ne_bytes)
        }
        ImageTensorData::Int8(data) => Value::from_raw_bytes(
            data.into_iter().map(|value| value as u8).collect(),
            shape,
            DataType::Int8,
        )
        .map_err(Into::into),
        ImageTensorData::Uint8(data) => {
            Value::from_raw_bytes(data, shape, DataType::Uint8).map_err(Into::into)
        }
        ImageTensorData::Bool(data) => {
            Value::from_raw_bytes(data, shape, DataType::Bool).map_err(Into::into)
        }
    }
}

fn typed_image_bytes<T, const N: usize>(
    data: Vec<T>,
    shape: &[i64],
    dtype: DataType,
    to_bytes: impl Fn(T) -> [u8; N],
) -> anyhow::Result<Value> {
    let bytes = data.into_iter().flat_map(to_bytes).collect::<Vec<u8>>();
    Value::from_raw_bytes(bytes, shape, dtype).map_err(Into::into)
}

fn literal_shape(contract: &TensorContract) -> anyhow::Result<Vec<i64>> {
    let Some(shape) = &contract.shape else {
        if contract.rank == 0 {
            return Ok(Vec::new());
        }
        anyhow::bail!("literal workflow input requires a fully declared shape");
    };
    shape
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Fixed(value) => Ok(*value),
            // A scalar literal initializes an otherwise-unbound symbolic axis as a singleton.
            // Concrete request/component values can subsequently unify shared symbols.
            TensorDimension::Symbol(_) => Ok(1),
        })
        .collect()
}

fn shape_numel(shape: &[i64]) -> usize {
    shape.iter().map(|dimension| *dimension as usize).product()
}

fn validate_workflow_value(
    name: &str,
    value: &Value,
    contract: &TensorContract,
    symbols: &mut HashMap<String, i64>,
    dynamic_symbols: &std::collections::HashSet<String>,
) -> anyhow::Result<()> {
    let expected_dtype = match contract.dtype.as_str() {
        "float32" | "fp32" => DataType::Float32,
        "float16" | "fp16" => DataType::Float16,
        "bfloat16" | "bf16" => DataType::BFloat16,
        "int64" => DataType::Int64,
        "int32" => DataType::Int32,
        "int16" => DataType::Int16,
        "int8" => DataType::Int8,
        "uint64" => DataType::Uint64,
        "uint32" => DataType::Uint32,
        "uint16" => DataType::Uint16,
        "uint8" => DataType::Uint8,
        "bool" => DataType::Bool,
        dtype => anyhow::bail!("workflow value '{name}' uses unsupported dtype '{dtype}'"),
    };
    if value.dtype() != expected_dtype {
        anyhow::bail!(
            "workflow value '{name}' has dtype {:?}, expected {}",
            value.dtype(),
            contract.dtype
        );
    }
    if value.shape().len() != contract.rank {
        anyhow::bail!(
            "workflow value '{name}' has rank {}, expected {}",
            value.shape().len(),
            contract.rank
        );
    }
    if let Some(shape) = &contract.shape {
        for (axis, (declared, actual)) in shape.iter().zip(value.shape()).enumerate() {
            match declared {
                TensorDimension::Fixed(expected) if expected != actual => anyhow::bail!(
                    "workflow value '{name}' axis {axis} is {actual}, expected {expected}"
                ),
                TensorDimension::Symbol(symbol) if dynamic_symbols.contains(symbol) => {}
                TensorDimension::Symbol(symbol) => match symbols.get(symbol) {
                    Some(expected) if expected != actual => anyhow::bail!(
                        "workflow value '{name}' axis {axis} binds symbol '{symbol}' to {actual}, \
                         but it was already {expected}"
                    ),
                    Some(_) => {}
                    None => {
                        symbols.insert(symbol.clone(), *actual);
                    }
                },
                TensorDimension::Fixed(_) => {}
            }
        }
    }
    Ok(())
}

fn validate_state_recurrence(
    cell: &str,
    current: &Value,
    next: &Value,
    state: &onnx_genai_metadata::WorkflowStateCell,
    values: &PipelineTensors,
) -> anyhow::Result<()> {
    if current.dtype() != next.dtype() || current.shape().len() != next.shape().len() {
        anyhow::bail!("workflow state '{cell}' update must preserve dtype and rank");
    }
    match &state.recurrence {
        onnx_genai_metadata::ShapeRecurrence::Invariant => {
            if current.shape() != next.shape() {
                anyhow::bail!(
                    "workflow state '{cell}' is invariant but changed shape from {:?} to {:?}",
                    current.shape(),
                    next.shape()
                );
            }
        }
        onnx_genai_metadata::ShapeRecurrence::Growing {
            axis,
            increment,
            max,
        } => {
            for (index, (before, after)) in current.shape().iter().zip(next.shape()).enumerate() {
                if index != *axis && before != after {
                    anyhow::bail!(
                        "workflow state '{cell}' changed non-growing axis {index} from {before} \
                         to {after}"
                    );
                }
            }
            let growth = i64::try_from(workflow_scalar_usize(values, increment)?)
                .context("workflow state growth increment exceeds i64")?;
            let limit = i64::try_from(workflow_scalar_usize(values, max)?)
                .context("workflow state growth limit exceeds i64")?;
            let before = *current.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' grows outside its tensor rank")
            })?;
            let after = *next.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' grows outside its tensor rank")
            })?;
            let expected = before
                .checked_add(growth)
                .with_context(|| format!("workflow state '{cell}' shape growth overflowed"))?;
            if after != expected {
                anyhow::bail!(
                    "workflow state '{cell}' growing axis {axis} changed from {before} to {after}, \
                     expected {expected}"
                );
            }
            if after > limit {
                anyhow::bail!(
                    "workflow state '{cell}' growing axis {axis} reached {after}, above maximum \
                     {limit}"
                );
            }
        }
        onnx_genai_metadata::ShapeRecurrence::Bounded { axis, max } => {
            for (index, (before, after)) in current.shape().iter().zip(next.shape()).enumerate() {
                if index != *axis && before != after {
                    anyhow::bail!(
                        "workflow state '{cell}' changed non-bounded axis {index} from {before} \
                         to {after}"
                    );
                }
            }
            let limit = i64::try_from(workflow_scalar_usize(values, max)?)
                .context("workflow state bounded-axis limit exceeds i64")?;
            let before = *current.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' bounded axis is outside its tensor rank")
            })?;
            let after = *next.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' bounded axis is outside its tensor rank")
            })?;
            if before > limit || after > limit {
                anyhow::bail!(
                    "workflow state '{cell}' bounded axis {axis} changed from {before} to {after}, \
                     above maximum {limit}"
                );
            }
        }
    }
    Ok(())
}

fn clone_pipeline_tensors(values: &PipelineTensors) -> anyhow::Result<PipelineTensors> {
    values
        .iter()
        .map(|(name, value)| Ok((name.clone(), clone_value(value)?)))
        .collect()
}

fn workflow_emitted_outputs(node: &WorkflowNode) -> std::collections::HashSet<String> {
    fn collect(node: &WorkflowNode, outputs: &mut std::collections::HashSet<String>) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect(node, outputs);
                }
            }
            WorkflowNode::Loop { setup, body, .. } => {
                collect(setup, outputs);
                collect(body, outputs);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect(case, outputs);
                }
                if let Some(default) = default {
                    collect(default, outputs);
                }
            }
            WorkflowNode::Emit { output, .. } => {
                outputs.insert(output.clone());
            }
            WorkflowNode::Invoke { .. } | WorkflowNode::Transfer { .. } => {}
        }
    }
    let mut outputs = std::collections::HashSet::new();
    collect(node, &mut outputs);
    outputs
}

fn append_workflow_value(previous: &Value, next: &Value) -> anyhow::Result<Value> {
    if previous.dtype() != next.dtype() || previous.shape().len() != next.shape().len() {
        anyhow::bail!("workflow append emit requires matching dtype and rank");
    }
    let mut shape = previous.shape().to_vec();
    let Some(last) = shape.last_mut() else {
        anyhow::bail!("workflow append emit requires rank >= 1");
    };
    for (left, right) in previous
        .shape()
        .iter()
        .zip(next.shape())
        .take(previous.shape().len() - 1)
    {
        if left != right {
            anyhow::bail!("workflow append emit requires equal non-appended dimensions");
        }
    }
    let left_width = *last as usize;
    let right_width = next.shape().last().copied().unwrap_or_default() as usize;
    let outer = previous.shape()[..previous.shape().len() - 1]
        .iter()
        .map(|dimension| *dimension as usize)
        .product::<usize>();
    *last += right_width as i64;
    let dtype = previous.dtype();
    let element_size = dtype.size_of();
    let left = previous.to_raw_bytes()?;
    let right = next.to_raw_bytes()?;
    let mut data = Vec::with_capacity(left.len() + right.len());
    for row in 0..outer {
        data.extend_from_slice(
            &left[row * left_width * element_size..(row + 1) * left_width * element_size],
        );
        data.extend_from_slice(
            &right[row * right_width * element_size..(row + 1) * right_width * element_size],
        );
    }
    Value::from_raw_bytes(data, &shape, dtype).map_err(Into::into)
}

fn emit_chunk_contract(
    output: &onnx_genai_metadata::TensorContract,
    emitted: &Value,
) -> anyhow::Result<onnx_genai_metadata::TensorContract> {
    let mut contract = output.clone();
    if let Some(shape) = &mut contract.shape {
        let dimension = shape
            .last_mut()
            .context("workflow prefix/append emit requires output rank >= 1")?;
        *dimension = TensorDimension::Fixed(
            *emitted
                .shape()
                .last()
                .context("workflow prefix/append emit requires value rank >= 1")?,
        );
    }
    Ok(contract)
}

fn slice_workflow_prefix(value: &Value, valid_length: usize) -> anyhow::Result<Value> {
    let mut shape = value.shape().to_vec();
    let rank = shape.len();
    let width = shape
        .last()
        .context("workflow emit valid_length requires a value with rank >= 1")?;
    let available = usize::try_from(*width).context("workflow emit has a negative final axis")?;
    if valid_length > available {
        anyhow::bail!(
            "workflow emit valid_length {valid_length} exceeds final-axis extent {available}"
        );
    }
    let outer = shape[..rank - 1]
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .context("workflow emit has a negative dimension")?
        .into_iter()
        .product::<usize>();
    let dtype = value.dtype();
    let element_size = dtype.size_of();
    let source = value.to_raw_bytes()?;
    let mut data = Vec::with_capacity(outer * valid_length * element_size);
    for row in 0..outer {
        let start = row * available * element_size;
        data.extend_from_slice(&source[start..start + valid_length * element_size]);
    }
    shape[rank - 1] =
        i64::try_from(valid_length).context("workflow emit valid_length exceeds i64")?;
    Value::from_raw_bytes(data, &shape, dtype).map_err(Into::into)
}

fn workflow_scalar_usize(values: &PipelineTensors, name: &str) -> anyhow::Result<usize> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow scalar value '{name}' is unavailable"))?;
    if value.shape().iter().try_fold(1usize, |size, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|dimension| size.checked_mul(dimension))
    }) != Some(1)
    {
        anyhow::bail!("workflow scalar '{name}' must contain exactly one value");
    }
    let data = value.to_raw_bytes()?;
    let signed = |value: i128| {
        usize::try_from(value)
            .with_context(|| format!("workflow scalar '{name}' must be non-negative"))
    };
    let unsigned = |value: u128| {
        usize::try_from(value).with_context(|| format!("workflow scalar '{name}' exceeds usize"))
    };
    match value.dtype() {
        DataType::Int8 => signed(i8::from_ne_bytes([data[0]]) as i128),
        DataType::Int16 => signed(i16::from_ne_bytes(data[..2].try_into()?) as i128),
        DataType::Int32 => signed(i32::from_ne_bytes(data[..4].try_into()?) as i128),
        DataType::Int64 => signed(i64::from_ne_bytes(data[..8].try_into()?) as i128),
        DataType::Uint8 => unsigned(u8::from_ne_bytes([data[0]]) as u128),
        DataType::Uint16 => unsigned(u16::from_ne_bytes(data[..2].try_into()?) as u128),
        DataType::Uint32 => unsigned(u32::from_ne_bytes(data[..4].try_into()?) as u128),
        DataType::Uint64 => unsigned(u64::from_ne_bytes(data[..8].try_into()?) as u128),
        dtype => {
            anyhow::bail!("workflow scalar '{name}' must have an integer dtype, got {dtype:?}")
        }
    }
}

fn workflow_scalar_bool(values: &PipelineTensors, name: &str) -> anyhow::Result<bool> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow predicate value '{name}' is unavailable"))?;
    match value.dtype() {
        DataType::Bool => {
            let data = value.to_raw_bytes()?;
            let [scalar] = data.as_slice() else {
                anyhow::bail!("workflow bool predicate '{name}' must contain exactly one value");
            };
            Ok(*scalar != 0)
        }
        _ => {
            let data = value.to_vec_i64()?;
            let [scalar] = data.as_slice() else {
                anyhow::bail!("workflow integer predicate '{name}' must contain exactly one value");
            };
            Ok(*scalar != 0)
        }
    }
}

fn workflow_scalar_key(values: &PipelineTensors, name: &str) -> anyhow::Result<String> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow branch value '{name}' is unavailable"))?;
    if value.dtype() == DataType::Bool {
        return Ok(workflow_scalar_bool(values, name)?.to_string());
    }
    let data = value.to_vec_i64()?;
    let [scalar] = data.as_slice() else {
        anyhow::bail!("workflow branch tensor '{name}' must contain exactly one value");
    };
    Ok(scalar.to_string())
}

#[cfg(test)]
mod workflow_scalar_tests {
    use super::*;

    #[test]
    fn batched_predicate_is_not_silently_reduced() {
        let mut values = PipelineTensors::new();
        values.insert(
            "done".to_string(),
            Value::from_raw_bytes(vec![0, 1], &[2], DataType::Bool).expect("bool tensor"),
        );

        let error = workflow_scalar_bool(&values, "done").expect_err("batched predicate fails");
        assert!(error.to_string().contains("exactly one value"));
    }

    #[test]
    fn batched_branch_key_is_not_silently_reduced() {
        let mut values = PipelineTensors::new();
        values.insert(
            "case".to_string(),
            Value::from_slice_i64(&[0, 1], &[2]).expect("integer tensor"),
        );

        let error = workflow_scalar_key(&values, "case").expect_err("batched branch key fails");
        assert!(error.to_string().contains("exactly one value"));
    }

    #[test]
    fn growing_state_uses_recurrence_instead_of_freezing_its_symbol() {
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, sequence] }")
                .expect("contract");
        let mut symbols = HashMap::new();
        let dynamic_symbols = std::collections::HashSet::from(["sequence".to_string()]);
        let current = Value::from_slice_i64(&[1, 2], &[1, 2]).expect("current");
        let next = Value::from_slice_i64(&[1, 2, 3], &[1, 3]).expect("next");
        validate_workflow_value(
            "current",
            &current,
            &contract,
            &mut symbols,
            &dynamic_symbols,
        )
        .expect("current contract");
        validate_workflow_value("next", &next, &contract, &mut symbols, &dynamic_symbols)
            .expect("growing symbol remains dynamic");

        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            r#"
contract: { dtype: int64, rank: 2, shape: [batch, sequence] }
scope: invocation
initializer: initial
recurrence: { kind: growing, axis: 1, increment: accepted, max: max_context }
"#,
        )
        .expect("state");
        let mut values = PipelineTensors::new();
        values.insert(
            "accepted".to_string(),
            Value::from_slice_i64(&[1], &[]).expect("increment"),
        );
        values.insert(
            "max_context".to_string(),
            Value::from_slice_i64(&[4], &[]).expect("limit"),
        );
        validate_state_recurrence("tokens", &current, &next, &state, &values)
            .expect("bounded growth validates");

        let invalid = Value::from_slice_i64(&[1, 2, 3, 4], &[1, 4]).expect("invalid next");
        let error = validate_state_recurrence("tokens", &current, &invalid, &state, &values)
            .expect_err("wrong increment fails");
        assert!(error.to_string().contains("expected 3"));
    }

    #[test]
    fn literals_and_append_support_declared_runtime_dtypes() {
        let int_contract: TensorContract =
            serde_yaml::from_str("{ dtype: int16, rank: 1, shape: [2] }").expect("contract");
        let integer =
            workflow_literal_value(&ScalarValue::Integer(7), &int_contract).expect("int16 literal");
        assert_eq!(integer.dtype(), DataType::Int16);

        let half_contract: TensorContract =
            serde_yaml::from_str("{ dtype: float16, rank: 1, shape: [2] }").expect("contract");
        let left =
            workflow_literal_value(&ScalarValue::Float(1.0), &half_contract).expect("half literal");
        let right =
            workflow_literal_value(&ScalarValue::Float(2.0), &half_contract).expect("half literal");
        let appended = append_workflow_value(&left, &right).expect("half append");
        assert_eq!(appended.dtype(), DataType::Float16);
        assert_eq!(appended.shape(), &[4]);
    }
}
