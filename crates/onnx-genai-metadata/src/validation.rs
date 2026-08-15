//! Validate metadata against runtime capabilities.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{InferenceMetadata, PipelineSpec, WorkflowNode, WorkflowSpec};

struct ContractObligation {
    id: &'static str,
    version: &'static str,
    action: &'static str,
    inputs: &'static [&'static str],
    outputs: &'static [&'static str],
}

const CONTRACT_OBLIGATIONS: &[ContractObligation] = &[
    ContractObligation {
        id: "onnx-genai.grammar-guidance",
        version: "1",
        action: "clone",
        inputs: &["state"],
        outputs: &["next_state"],
    },
    ContractObligation {
        id: "onnx-genai.grammar-guidance",
        version: "1",
        action: "lookahead",
        inputs: &["state", "tokens", "valid_length", "transition_table"],
        outputs: &[
            "next_state",
            "consumed_length",
            "logits_mask",
            "forced_tokens",
            "forced_length",
        ],
    },
    ContractObligation {
        id: "onnx-genai.grammar-guidance",
        version: "1",
        action: "commit",
        inputs: &["state", "tokens", "valid_length", "transition_table"],
        outputs: &["next_state", "consumed_length"],
    },
    ContractObligation {
        id: "onnx-genai.telemetry",
        version: "1",
        action: "start",
        inputs: &[],
        outputs: &["timestamp"],
    },
    ContractObligation {
        id: "onnx-genai.telemetry",
        version: "1",
        action: "elapsed",
        inputs: &["timestamp"],
        outputs: &["duration_ms"],
    },
    ContractObligation {
        id: "onnx-genai.parameter-overlay",
        version: "1",
        action: "apply",
        inputs: &["input"],
        outputs: &["output"],
    },
];

/// Capabilities this runtime supports.
pub struct RuntimeCapabilities {
    pub supported: Vec<String>,
}

impl Default for RuntimeCapabilities {
    fn default() -> Self {
        Self {
            supported: vec![
                "kv_cache".to_string(),
                "grouped_query_attention".to_string(),
                "multi_head_attention".to_string(),
                "prefix_cache".to_string(),
                "continuous_batching".to_string(),
                "control_flow_loop".to_string(),
            ],
        }
    }
}

/// Validate the metadata document and required runtime capabilities.
pub fn validate(
    metadata: &InferenceMetadata,
    runtime: &RuntimeCapabilities,
) -> Result<(), Vec<String>> {
    let mut errors = validate_metadata(metadata).err().unwrap_or_default();
    let required = metadata
        .required_capabilities
        .iter()
        .cloned()
        .chain(derived_capabilities(metadata));
    errors.extend(required.filter(|capability| !runtime.supported.contains(capability)));

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Capabilities implied by concrete metadata features.
pub fn derived_capabilities(metadata: &InferenceMetadata) -> BTreeSet<String> {
    let mut capabilities = BTreeSet::new();
    let Some(pipeline) = &metadata.pipeline else {
        return capabilities;
    };
    let workflow = &pipeline.workflow;
    capabilities.extend(workflow.manifest.capabilities.iter().cloned());
    capabilities.insert("workflow_ssa".to_string());
    if workflow.serving.is_some() {
        capabilities.insert("serving_service_contract".to_string());
    }
    if workflow.adapters.is_some() {
        capabilities.insert("parameter_adapters".to_string());
        capabilities.insert("heterogeneous_adapter_batching".to_string());
    }
    if workflow
        .state
        .values()
        .any(|state| state.scope == crate::schema::WorkflowStateScope::Session)
    {
        capabilities.insert("session_state_lease".to_string());
    }
    if workflow.state.values().any(|state| {
        matches!(
            state.recurrence,
            crate::schema::ShapeRecurrence::Bounded { .. }
        )
    }) {
        capabilities.insert("bounded_state_recurrence".to_string());
    }
    if workflow
        .state
        .values()
        .any(|state| state.class == crate::schema::WorkflowStateClass::Advisory)
    {
        capabilities.insert("advisory_state".to_string());
    }
    for component in workflow.components.values() {
        match component
            .contract
            .as_ref()
            .map(|contract| contract.id.as_str())
        {
            Some("onnx-genai.adaptive-proposal-budget") => {
                capabilities.insert("adaptive_proposal_budget".to_string());
            }
            Some("onnx-genai.grammar-guidance") => {
                capabilities.insert("grammar_guidance_adapter".to_string());
            }
            Some("onnx-genai.telemetry") => {
                capabilities.insert("telemetry_adapter".to_string());
            }
            _ => {}
        }
    }
    if let Ok(compiled) = crate::compile_workflow(workflow) {
        collect_workflow_capabilities(&compiled.graph, &mut capabilities);
    }
    capabilities
}

fn collect_workflow_capabilities(node: &WorkflowNode, capabilities: &mut BTreeSet<String>) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for node in nodes {
                collect_workflow_capabilities(node, capabilities);
            }
        }
        WorkflowNode::Invoke { .. } => {}
        WorkflowNode::Loop {
            setup,
            body,
            iteration,
            ..
        } => {
            capabilities.insert("nested_control_flow".to_string());
            if iteration.is_some() {
                capabilities.insert("loop_induction_values".to_string());
            }
            collect_workflow_capabilities(setup, capabilities);
            collect_workflow_capabilities(body, capabilities);
        }
        WorkflowNode::Branch { cases, default, .. } => {
            capabilities.insert("nested_control_flow".to_string());
            for case in cases.values() {
                collect_workflow_capabilities(case, capabilities);
            }
            if let Some(default) = default {
                collect_workflow_capabilities(default, capabilities);
            }
        }
        WorkflowNode::Emit {
            mode,
            valid_length,
            row_ids,
            ..
        } => {
            capabilities.insert("typed_emit".to_string());
            if matches!(mode, crate::schema::WorkflowEmitMode::Event) {
                capabilities.insert("streaming_emit".to_string());
            }
            if valid_length.is_some() {
                capabilities.insert("emit_valid_length".to_string());
            }
            if row_ids.is_some() {
                capabilities.insert("emit_row_identity".to_string());
            }
        }
        WorkflowNode::Transfer { .. } => {
            capabilities.insert("explicit_transfer".to_string());
        }
        WorkflowNode::ExecutionIsland { .. } => {}
    }
}

/// Validate document-level invariants independent of runtime capabilities.
pub fn validate_metadata(metadata: &InferenceMetadata) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if let Err(error) = validate_composite_io(metadata) {
        errors.push(error);
    }

    if let Some(pipeline) = &metadata.pipeline
        && let Err(error) = validate_pipeline_spec(pipeline)
    {
        errors.extend(error.errors);
    }
    validate_preprocessing_workflow(metadata, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_preprocessing_workflow(metadata: &InferenceMetadata, errors: &mut Vec<String>) {
    const IMAGE_ABI: &str = "onnx-genai.image-preprocess";
    const IMAGE_ABI_VERSION: &str = "1";

    let Some(workflow) = metadata
        .pipeline
        .as_ref()
        .map(|pipeline| &pipeline.workflow)
    else {
        return;
    };
    let image = metadata
        .preprocessing
        .as_ref()
        .and_then(|spec| spec.image.as_ref());
    let adapters = workflow
        .components
        .iter()
        .filter(|(_, component)| {
            matches!(
                &component.implementation,
                crate::schema::ComponentImplementation::Adapter { abi, version, .. }
                    if abi == IMAGE_ABI && version == IMAGE_ABI_VERSION
            )
        })
        .collect::<Vec<_>>();
    if image.is_none() && !adapters.is_empty() {
        errors.push(format!(
            "workflow adapter components using {IMAGE_ABI}@{IMAGE_ABI_VERSION} require \
                 preprocessing.image metadata"
        ));
        return;
    }
    let Some(image) = image else {
        return;
    };
    if adapters.len() != 1 {
        errors.push(format!(
            "preprocessing.image requires exactly one workflow adapter component using \
                 {IMAGE_ABI}@{IMAGE_ABI_VERSION}, found {}",
            adapters.len()
        ));
        return;
    }
    let (adapter_name, adapter) = adapters[0];
    match adapter.ports.inputs.get("encoded") {
        Some(contract) if contract.dtype == "uint8" && contract.rank == 1 => {}
        Some(contract) => errors.push(format!(
            "workflow image preprocessing adapter '{adapter_name}' input 'encoded' must be uint8 \
                 rank 1, got {} rank {}",
            contract.dtype, contract.rank
        )),
        None => errors.push(format!(
            "workflow image preprocessing adapter '{adapter_name}' must declare input 'encoded'"
        )),
    }

    fn collect_invocations<'a>(
        node: &'a WorkflowNode,
        component: &str,
        outputs: &mut Vec<&'a BTreeMap<String, String>>,
    ) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    collect_invocations(node, component, outputs);
                }
            }
            WorkflowNode::Invoke {
                component: invoked,
                outputs: invoked_outputs,
                ..
            } if invoked == component => outputs.push(invoked_outputs),
            WorkflowNode::Loop { setup, body, .. } => {
                collect_invocations(setup, component, outputs);
                collect_invocations(body, component, outputs);
            }
            WorkflowNode::Branch { cases, default, .. } => {
                for case in cases.values() {
                    collect_invocations(case, component, outputs);
                }
                if let Some(default) = default {
                    collect_invocations(default, component, outputs);
                }
            }
            WorkflowNode::Invoke { .. }
            | WorkflowNode::Emit { .. }
            | WorkflowNode::Transfer { .. }
            | WorkflowNode::ExecutionIsland { .. } => {}
        }
    }
    let Ok(compiled) = crate::compile_workflow(workflow) else {
        return;
    };
    let mut invocations = Vec::new();
    collect_invocations(&compiled.graph, adapter_name, &mut invocations);
    if invocations.len() != 1 {
        errors.push(format!(
            "workflow image preprocessing adapter '{adapter_name}' must be invoked exactly once, \
                 found {} invocations",
            invocations.len()
        ));
        return;
    }
    let invocation_outputs = invocations[0];
    for output in &image.outputs {
        if output.optional.unwrap_or(false) {
            errors.push(format!(
                "preprocessing.image output '{}' cannot be optional in a workflow; every declared \
                 adapter SSA output must be materialized",
                output.name
            ));
        }
        let Some(contract) = &output.contract else {
            errors.push(format!(
                "preprocessing.image output '{}' must declare a TensorContract for workflow use",
                output.name
            ));
            continue;
        };
        if contract.dtype != output.dtype {
            errors.push(format!(
                "preprocessing.image output '{}' dtype '{}' disagrees with its TensorContract '{}'",
                output.name, output.dtype, contract.dtype
            ));
        }
        let port = invocation_outputs
            .iter()
            .find_map(|(port, value)| (value == &output.name).then_some(port));
        let Some(port) = port else {
            errors.push(format!(
                "preprocessing.image output '{}' must be a declared SSA output of adapter \
                     invocation '{adapter_name}'",
                output.name
            ));
            continue;
        };
        match adapter.ports.outputs.get(port) {
            Some(port_contract) => require_compatible_tensor_contracts(
                contract,
                port_contract,
                &format!("preprocessing.image output '{}'", output.name),
                errors,
            ),
            None => errors.push(format!(
                "workflow image preprocessing adapter '{adapter_name}' has no output port '{port}'"
            )),
        }
    }
}

fn require_compatible_tensor_contracts(
    source: &crate::schema::TensorContract,
    target: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    fn normalize(dtype: &str) -> &str {
        match dtype {
            "fp32" => "float32",
            "fp16" => "float16",
            "bf16" => "bfloat16",
            other => other,
        }
    }
    if normalize(&source.dtype) != normalize(&target.dtype)
        || source.rank != target.rank
        || source.shape != target.shape
    {
        errors.push(format!(
            "{path} has a contract incompatible with its adapter output port"
        ));
    }
}

pub(crate) fn validate_composite_io(metadata: &InferenceMetadata) -> Result<(), String> {
    if metadata.pipeline.is_some()
        && metadata
            .model
            .as_ref()
            .and_then(|model| model.io.as_ref())
            .is_some()
    {
        Err(
            "model.io is only valid for bare single-model metadata; when pipeline is present, \
             declare decoder I/O at pipeline.models.<component>.io"
                .to_string(),
        )
    } else {
        Ok(())
    }
}

/// All structural problems found in a pipeline specification.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid pipeline spec: {errors:?}")]
pub struct PipelineValidationError {
    pub errors: Vec<String>,
}

/// Validate the pipeline DAG and component references.
pub fn validate_pipeline_spec(spec: &PipelineSpec) -> Result<(), PipelineValidationError> {
    let mut errors = Vec::new();
    validate_workflow(&spec.workflow, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(PipelineValidationError { errors })
    }
}

fn valid_adapter_base_fingerprint(value: &str) -> bool {
    let Some(digest) = value.strip_prefix("onnx-genai-targeted-base-v1:sha256:") else {
        return false;
    };
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[allow(clippy::too_many_arguments)]
fn validate_adapter_selection_input(
    workflow: &WorkflowSpec,
    name: &str,
    dtype: &str,
    rank: usize,
    second_dimension: Option<usize>,
    expected_role: crate::schema::RuntimeInputRole,
    field: &str,
    errors: &mut Vec<String>,
) {
    let expected_shape = |shape: &[crate::schema::TensorDimension]| match second_dimension {
        None => matches!(
            shape,
            [crate::schema::TensorDimension::Symbol(symbol)] if symbol == "batch"
        ),
        Some(extent) => matches!(
            shape,
            [
                crate::schema::TensorDimension::Symbol(symbol),
                crate::schema::TensorDimension::Fixed(actual)
            ] if symbol == "batch" && *actual == extent as i64
        ),
    };
    match workflow.inputs.get(name) {
        Some(input)
            if input.contract.dtype == dtype
                && input.contract.rank == rank
                && input
                    .contract
                    .shape
                    .as_deref()
                    .is_some_and(expected_shape)
                && input.required
                && input.source == crate::schema::WorkflowInputSource::Request
                && matches!(
                    &input.role,
                    crate::schema::SemanticInputRole::Runtime { version, role }
                        if version == "1.0" && role == &expected_role
                ) => {}
        Some(_) => errors.push(format!(
            "pipeline.workflow.adapters.selection.{field} '{name}' must reference a required \
             request-sourced {dtype}{} workflow input with runtime role {:?}@1.0",
            if let Some(extent) = second_dimension {
                format!("[batch,{extent}]")
            } else {
                "[batch]".to_string()
            },
            expected_role
        )),
        None => errors.push(format!(
            "pipeline.workflow.adapters.selection.{field} '{name}' references an undeclared workflow input"
        )),
    }
}

fn validate_adapter_service(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    let Some(service) = &workflow.adapters else {
        return;
    };
    if !valid_adapter_base_fingerprint(&service.base_model_fingerprint) {
        errors.push(
            "pipeline.workflow.adapters.base_model_fingerprint must be \
             onnx-genai-targeted-base-v1:sha256:<64 lowercase hexadecimal characters>"
                .into(),
        );
    }
    if service.application_capability.trim().is_empty() {
        errors.push("pipeline.workflow.adapters.application_capability must not be empty".into());
    } else if service.application_capability != "onnx-genai.adapters@1" {
        errors.push(
            "pipeline.workflow.adapters.application_capability must be onnx-genai.adapters@1"
                .into(),
        );
    }
    if service.cache.max_entries == 0 {
        errors
            .push("pipeline.workflow.adapters.cache.max_entries must be greater than zero".into());
    }
    if service.selection.max_adapters == 0 {
        errors.push(
            "pipeline.workflow.adapters.selection.max_adapters must be greater than zero".into(),
        );
    }
    if service.artifacts.is_empty() {
        errors.push("pipeline.workflow.adapters.artifacts must not be empty".into());
    }
    validate_adapter_selection_input(
        workflow,
        &service.selection.row_ids,
        "int64",
        1,
        None,
        crate::schema::RuntimeInputRole::RowIds,
        "row_ids",
        errors,
    );
    validate_adapter_selection_input(
        workflow,
        &service.selection.request_epochs,
        "int64",
        1,
        None,
        crate::schema::RuntimeInputRole::RequestEpochs,
        "request_epochs",
        errors,
    );
    validate_adapter_selection_input(
        workflow,
        &service.selection.adapter_ids,
        "int64",
        2,
        Some(service.selection.max_adapters),
        crate::schema::RuntimeInputRole::AdapterIds,
        "adapter_ids",
        errors,
    );
    validate_adapter_selection_input(
        workflow,
        &service.selection.adapter_counts,
        "int64",
        1,
        None,
        crate::schema::RuntimeInputRole::AdapterCounts,
        "adapter_counts",
        errors,
    );
    validate_adapter_selection_input(
        workflow,
        &service.selection.scales,
        "float32",
        2,
        Some(service.selection.max_adapters),
        crate::schema::RuntimeInputRole::AdapterScales,
        "scales",
        errors,
    );
    if let Some(active) = &service.selection.active {
        validate_adapter_selection_input(
            workflow,
            active,
            "bool",
            1,
            None,
            crate::schema::RuntimeInputRole::AdapterActive,
            "active",
            errors,
        );
    }
    let mut identities = BTreeSet::new();
    let mut indices = BTreeSet::new();
    let mut target_owners = BTreeMap::<(String, String), String>::new();
    for (name, artifact) in &service.artifacts {
        let path = format!("pipeline.workflow.adapters.artifacts.{name}");
        if artifact.identity.trim().is_empty() || artifact.version.trim().is_empty() {
            errors.push(format!("{path} identity and version must not be empty"));
        }
        if !indices.insert(artifact.index) {
            errors.push(format!(
                "{path}.index {} duplicates another adapter wire ID",
                artifact.index
            ));
        }
        if !identities.insert((artifact.identity.clone(), artifact.version.clone())) {
            errors.push(format!(
                "{path} duplicates adapter identity {}@{}",
                artifact.identity, artifact.version
            ));
        }
        if artifact.base_model_fingerprint != service.base_model_fingerprint {
            errors.push(format!(
                "{path}.base_model_fingerprint '{}' does not match service fingerprint '{}'",
                artifact.base_model_fingerprint, service.base_model_fingerprint
            ));
        }
        if artifact.rank == 0 {
            errors.push(format!("{path}.rank must be greater than zero"));
        }
        if !artifact.alpha.is_finite() || artifact.alpha <= 0.0 {
            errors.push(format!("{path}.alpha must be finite and greater than zero"));
        }
        if !matches!(
            artifact.dtype.as_str(),
            "float16" | "fp16" | "float32" | "fp32" | "bfloat16" | "bf16"
        ) {
            errors.push(format!(
                "{path}.dtype '{}' must be a floating-point adapter dtype",
                artifact.dtype
            ));
        }
        if artifact.weights.is_empty() {
            errors.push(format!(
                "{path}.weights must declare at least one external artifact"
            ));
        }
        let mut weight_formats = BTreeSet::new();
        for (index, weight) in artifact.weights.iter().enumerate() {
            if !weight_formats.insert(weight.format.clone()) {
                errors.push(format!(
                    "{path}.weights contains duplicate format {:?}",
                    weight.format
                ));
            }
            if weight.location.trim().is_empty()
                || std::path::Path::new(&weight.location).is_absolute()
                || !weight.location.starts_with(&format!("adapters/{name}/"))
                || weight
                    .location
                    .split(['/', '\\'])
                    .any(|segment| segment == "..")
            {
                errors.push(format!(
                    "{path}.weights[{index}].location must be under package path adapters/{name}/"
                ));
            }
            if weight.sha256.len() != 64
                || !weight
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                errors.push(format!(
                    "{path}.weights[{index}].sha256 must be 64 lowercase hexadecimal characters"
                ));
            }
        }
        if artifact.targets.is_empty() {
            errors.push(format!("{path}.targets must not be empty"));
        }
        let mut local_targets = BTreeSet::new();
        let mut local_weight_keys = BTreeSet::new();
        let has_ort_bundle = artifact
            .weights
            .iter()
            .any(|weight| weight.format == crate::schema::AdapterWeightFormat::OrtGenai);
        for (index, target) in artifact.targets.iter().enumerate() {
            let target_path = format!("{path}.targets[{index}]");
            if !workflow.components.contains_key(&target.component) {
                errors.push(format!(
                    "{target_path}.component '{}' is undeclared",
                    target.component
                ));
            }
            if target.parameter.trim().is_empty() || target.weight_key.trim().is_empty() {
                errors.push(format!(
                    "{target_path} parameter and weight_key must not be empty"
                ));
            }
            if !local_weight_keys.insert(target.weight_key.clone()) {
                errors.push(format!(
                    "{path} declares duplicate weight_key '{}'",
                    target.weight_key
                ));
            }
            match &target.native_parameters {
                Some(native)
                    if native.a.trim().is_empty()
                        || native.b.trim().is_empty()
                        || native.a == native.b =>
                {
                    errors.push(format!(
                        "{target_path}.native_parameters must contain distinct non-empty a and b initializer names"
                    ));
                }
                None if has_ort_bundle => errors.push(format!(
                    "{target_path}.native_parameters is required for an ort_genai weight artifact"
                )),
                _ => {}
            }
            if target.input_features == 0 || target.output_features == 0 {
                errors.push(format!(
                    "{target_path} input_features and output_features must be greater than zero"
                ));
            }
            let key = (target.component.clone(), target.parameter.clone());
            if !local_targets.insert(key.clone()) {
                errors.push(format!(
                    "{path} declares duplicate target '{}.{}'",
                    target.component, target.parameter
                ));
            }
            if let Some(owner) = target_owners.insert(key, name.clone())
                && owner != *name
            {
                // Multiple adapters may compose on one target, but the target dimensions must
                // remain identical. The cross-artifact check below enforces that contract.
                let owner_target = service
                    .artifacts
                    .get(&owner)
                    .into_iter()
                    .flat_map(|artifact| &artifact.targets)
                    .find(|candidate| {
                        candidate.component == target.component
                            && candidate.parameter == target.parameter
                    });
                if owner_target.is_some_and(|owner_target| {
                    owner_target.input_features != target.input_features
                        || owner_target.output_features != target.output_features
                        || owner_target.native_parameters != target.native_parameters
                }) {
                    errors.push(format!(
                        "{target_path} conflicts with adapter '{owner}' binding for '{}.{}'",
                        target.component, target.parameter
                    ));
                }
            }
        }
    }
    if indices
        .iter()
        .copied()
        .enumerate()
        .any(|(expected, actual)| expected != actual)
    {
        errors.push(format!(
            "pipeline.workflow.adapters artifact indices must be contiguous from zero; found {indices:?}"
        ));
    }
    for (name, component) in &workflow.components {
        let Some(contract) = component
            .contract
            .as_ref()
            .filter(|contract| contract.id == "onnx-genai.parameter-overlay")
        else {
            continue;
        };
        let string_parameter = |key: &str| match contract.parameters.get(key) {
            Some(crate::schema::ScalarValue::String(value)) if !value.trim().is_empty() => {
                Some(value.as_str())
            }
            _ => None,
        };
        let target_component = string_parameter("component");
        let target_parameter = string_parameter("parameter");
        if target_component.is_none() {
            errors.push(format!(
                "workflow parameter overlay component '{name}' must declare non-empty string parameter 'component'"
            ));
        }
        if target_parameter.is_none() {
            errors.push(format!(
                "workflow parameter overlay component '{name}' must declare non-empty string parameter 'parameter'"
            ));
        }
        if let (Some(target_component), Some(target_parameter)) =
            (target_component, target_parameter)
            && !target_owners
                .contains_key(&(target_component.to_string(), target_parameter.to_string()))
        {
            errors.push(format!(
                "workflow parameter overlay component '{name}' targets undeclared adapter parameter '{target_component}.{target_parameter}'"
            ));
        }
    }
}

fn validate_workflow(workflow: &WorkflowSpec, errors: &mut Vec<String>) {
    let compiled = match crate::compile_workflow(workflow) {
        Ok(compiled) => compiled,
        Err(error) => {
            errors.push(format!("pipeline.workflow lowering failed: {error}"));
            return;
        }
    };
    if workflow.manifest.ir_version != "1.0" {
        errors.push(format!(
            "unsupported pipeline.workflow.manifest.ir_version '{}'; this runtime supports 1.0",
            workflow.manifest.ir_version
        ));
    }
    if workflow.manifest.onnx_opsets.is_empty() {
        errors.push("pipeline.workflow.manifest.onnx_opsets must not be empty".to_string());
    }
    validate_adapter_service(workflow, errors);
    for (domain, version) in &workflow.manifest.onnx_opsets {
        if domain.trim().is_empty() || *version == 0 {
            errors.push(format!(
                "pipeline.workflow.manifest.onnx_opsets contains invalid {domain:?}@{version}"
            ));
        }
    }

    fn validate_runtime_dtype(
        path: &str,
        contract: &crate::schema::TensorContract,
        errors: &mut Vec<String>,
    ) {
        if contract
            .shape
            .as_ref()
            .is_some_and(|shape| shape.len() != contract.rank)
        {
            errors.push(format!(
                "{path} declares rank {} but has {} shape dimensions",
                contract.rank,
                contract.shape.as_ref().map_or(0, Vec::len)
            ));
        }
        if !matches!(
            contract.dtype.as_str(),
            "float16"
                | "fp16"
                | "float32"
                | "fp32"
                | "bfloat16"
                | "bf16"
                | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
                | "bool"
        ) {
            errors.push(format!(
                "{path} uses dtype '{}', which the workflow runtime does not support",
                contract.dtype
            ));
        }
    }

    let mut initial_value_names = workflow.inputs.keys().cloned().collect::<BTreeSet<_>>();
    for (name, input) in &workflow.inputs {
        validate_runtime_dtype(&format!("workflow input '{name}'"), &input.contract, errors);
        if matches!(input.source, crate::schema::WorkflowInputSource::Request)
            && !matches!(input.role, crate::schema::SemanticInputRole::Runtime { .. })
        {
            errors.push(format!(
                "workflow request input '{name}' must declare one versioned runtime role"
            ));
        }
        if !input.required && input.default.is_none() && input.present_as.is_none() {
            errors.push(format!(
                "workflow input '{name}' cannot be optional without a literal default or \
                 present_as predicate"
            ));
        }
        if let Some(present_as) = &input.present_as {
            if input.required {
                errors.push(format!(
                    "workflow input '{name}' uses present_as but is required; presence predicates \
                     are only valid for optional inputs"
                ));
            }
            if input.default.is_some() {
                errors.push(format!(
                    "workflow input '{name}' cannot combine present_as with a literal default"
                ));
            }
            if !matches!(
                input.source,
                crate::schema::WorkflowInputSource::Request
                    | crate::schema::WorkflowInputSource::Application { .. }
            ) {
                errors.push(format!(
                    "workflow input '{name}' can use present_as only with request or application \
                     sources"
                ));
            }
            if matches!(input.source, crate::schema::WorkflowInputSource::Request)
                && !matches!(
                    input.role,
                    crate::schema::SemanticInputRole::Runtime {
                        role: crate::schema::RuntimeInputRole::Media
                            | crate::schema::RuntimeInputRole::Constraint
                            | crate::schema::RuntimeInputRole::SessionId,
                        ..
                    }
                )
            {
                errors.push(format!(
                    "workflow request input '{name}' can use present_as only with media, \
                     constraint, or session_id roles whose absence is observable"
                ));
            }
            if present_as.trim().is_empty() {
                errors.push(format!(
                    "workflow input '{name}' has an empty present_as value"
                ));
            } else if !initial_value_names.insert(present_as.clone()) {
                errors.push(format!(
                    "workflow input '{name}' present_as value '{present_as}' collides with another \
                     initial SSA value"
                ));
            }
        }
    }
    for (name, input) in &workflow.inputs {
        if let Some(present_as) = &input.present_as {
            validate_optional_input_guards(
                &compiled.graph,
                name,
                present_as,
                false,
                "pipeline.workflow.steps",
                errors,
            );
        }
    }
    for (name, output) in &workflow.outputs {
        validate_runtime_dtype(
            &format!("workflow output '{name}'"),
            &output.contract,
            errors,
        );
    }
    for (name, component) in &workflow.components {
        if name.trim().is_empty() || name.contains('.') {
            errors.push(format!("workflow component name is invalid: '{name}'"));
        }
        match &component.implementation {
            crate::schema::ComponentImplementation::Onnx { artifact } => {
                if artifact.trim().is_empty() {
                    errors.push(format!(
                        "workflow component '{name}' has an empty ONNX artifact"
                    ));
                }
            }
            crate::schema::ComponentImplementation::Adapter {
                abi,
                version,
                artifact: _,
            } => match workflow.manifest.adapter_abis.get(abi) {
                Some(pinned) if pinned == version => {}
                _ => errors.push(format!(
                    "workflow component '{name}' requires adapter ABI {abi}@{version}, \
                         but the manifest does not pin that exact version"
                )),
            },
            crate::schema::ComponentImplementation::Binding => {}
        }
        for (port, contract) in &component.ports.inputs {
            validate_runtime_dtype(
                &format!("workflow component '{name}' input '{port}'"),
                contract,
                errors,
            );
        }
        for (port, contract) in &component.ports.outputs {
            validate_runtime_dtype(
                &format!("workflow component '{name}' output '{port}'"),
                contract,
                errors,
            );
        }
        if let Some(contract) = &component.contract {
            if contract.id.trim().is_empty() || contract.version.trim().is_empty() {
                errors.push(format!(
                    "workflow component '{name}' contract id and version must not be empty"
                ));
            }
            if !(component.ports.inputs.is_empty() && component.ports.outputs.is_empty()) {
                for (role, port) in &contract.bindings {
                    if role.trim().is_empty()
                        || (!component.ports.inputs.contains_key(port)
                            && !component.ports.outputs.contains_key(port))
                    {
                        errors.push(format!(
                            "workflow component '{name}' contract binding '{role}' references \
                             unknown port '{port}'"
                        ));
                    }
                }
            }
            if let crate::schema::ComponentImplementation::Adapter { abi, version, .. } =
                &component.implementation
            {
                if contract.id != *abi || contract.version != *version {
                    errors.push(format!(
                        "workflow adapter component '{name}' contract {}@{} must match its ABI \
                             {abi}@{version}",
                        contract.id, contract.version
                    ));
                }
                let action = match contract.parameters.get("action") {
                    Some(crate::schema::ScalarValue::String(action)) => Some(action.as_str()),
                    _ => None,
                };
                let known_contract = CONTRACT_OBLIGATIONS
                    .iter()
                    .any(|entry| entry.id == contract.id && entry.version == contract.version);
                let obligation = CONTRACT_OBLIGATIONS.iter().find(|entry| {
                    entry.id == contract.id
                        && entry.version == contract.version
                        && action == Some(entry.action)
                });
                if known_contract && obligation.is_none() {
                    errors.push(format!(
                        "workflow adapter component '{name}' has unsupported action {:?} for \
                             contract {}@{}",
                        action, contract.id, contract.version
                    ));
                }
                for role in obligation
                    .into_iter()
                    .flat_map(|entry| entry.inputs.iter().chain(entry.outputs))
                {
                    if !contract.bindings.contains_key(*role) {
                        errors.push(format!(
                            "workflow adapter component '{name}' contract {}@{} is missing \
                             required binding '{role}'",
                            contract.id, contract.version
                        ));
                    }
                }
                for (direction, roles, ports) in obligation.into_iter().flat_map(|entry| {
                    [
                        ("input", entry.inputs, &component.ports.inputs),
                        ("output", entry.outputs, &component.ports.outputs),
                    ]
                }) {
                    for role in roles {
                        if contract
                            .bindings
                            .get(*role)
                            .is_some_and(|port| !ports.contains_key(port))
                        {
                            errors.push(format!(
                                "workflow adapter component '{name}' contract role '{role}' must \
                                 bind an {direction} port"
                            ));
                        }
                    }
                }
            }
        }
        if component.application_overridable && component.contract.is_none() {
            errors.push(format!(
                "workflow component '{name}' is application-overridable but has no versioned contract"
            ));
        }
        if component.application_overridable
            && !matches!(
                component.implementation,
                crate::schema::ComponentImplementation::Onnx { .. }
            )
        {
            errors.push(format!(
                "workflow component '{name}' is application-overridable but is not an ONNX component"
            ));
        }
    }
    for (name, state) in &workflow.state {
        validate_runtime_dtype(&format!("workflow state '{name}'"), &state.contract, errors);
        if state.scope == crate::schema::WorkflowStateScope::Invocation && state.session.is_some() {
            errors.push(format!(
                "workflow state '{name}' has session lease settings but invocation scope"
            ));
        }
        if let Some(session) = &state.session {
            if session.policy != crate::schema::SessionMutationPolicy::Exclusive {
                errors.push(format!(
                    "workflow state '{name}' requests copy-on-write session mutation, \
                     which this runtime does not support"
                ));
            }
            if session.ttl_seconds.is_some() {
                errors.push(format!(
                    "workflow state '{name}' declares session TTL, which this runtime \
                     does not yet enforce"
                ));
            }
            if session.optimistic_metadata_version {
                errors.push(format!(
                    "workflow state '{name}' requests optimistic metadata versioning, \
                     which this runtime does not support"
                ));
            }
        }

        let dynamic_axis = match &state.recurrence {
            crate::schema::ShapeRecurrence::Growing { axis, .. }
            | crate::schema::ShapeRecurrence::Bounded { axis, .. } => Some(*axis),
            crate::schema::ShapeRecurrence::Invariant => None,
        };
        if let Some(axis) = dynamic_axis
            && axis >= state.contract.rank
        {
            errors.push(format!(
                "workflow state '{name}' varies on axis {axis}, outside rank {}",
                state.contract.rank
            ));
        }
        if let Some(group_name) = &state.service_group {
            let group = workflow
                .serving
                .as_ref()
                .and_then(|serving| serving.kv_service.groups.get(group_name));
            let Some(group) = group else {
                errors.push(format!(
                    "workflow state '{name}' binds unknown KV service group '{group_name}'"
                ));
                continue;
            };
            if group.sequence_axis >= state.contract.rank {
                errors.push(format!(
                    "KV service group '{group_name}' sequence_axis {} is outside state '{name}' rank {}",
                    group.sequence_axis, state.contract.rank
                ));
            }
            if dynamic_axis.is_some_and(|axis| axis != group.sequence_axis) {
                errors.push(format!(
                    "workflow state '{name}' recurrence axis {dynamic_axis:?} disagrees with KV \
                     service group '{group_name}' sequence_axis {}",
                    group.sequence_axis
                ));
            }
            if let crate::schema::ShapeRecurrence::Growing { increment, .. } = &state.recurrence
                && let Some(serving) = &workflow.serving
                && serving.accepted_len.as_deref() != Some(increment)
            {
                errors.push(format!(
                    "KV service state '{name}' grows by '{increment}', but serving.accepted_len \
                     does not bind that per-row value"
                ));
            }
        }
    }

    if let Some(kv_service) = workflow.serving.as_ref().map(|serving| &serving.kv_service) {
        for (group_name, group) in &kv_service.groups {
            if group.layout.trim().is_empty() {
                errors.push(format!(
                    "KV service group '{group_name}' layout must not be empty"
                ));
            }
            match workflow.state.get(&group.logical_lengths) {
                Some(lengths) => {
                    validate_integer_control_contract(
                        &lengths.contract,
                        &format!(
                            "KV service group '{group_name}' logical_lengths state '{}'",
                            group.logical_lengths
                        ),
                        errors,
                    );
                    if lengths.contract.rank != 1 {
                        errors.push(format!(
                            "KV service group '{group_name}' logical_lengths state '{}' must be \
                             rank one with one value per row",
                            group.logical_lengths
                        ));
                    }
                    if lengths.class != crate::schema::WorkflowStateClass::Semantic {
                        errors.push(format!(
                            "KV service group '{group_name}' logical_lengths state '{}' must be \
                             semantic for checkpoint/replay",
                            group.logical_lengths
                        ));
                    }
                }
                None => errors.push(format!(
                    "KV service group '{group_name}' references unknown logical_lengths state '{}'",
                    group.logical_lengths
                )),
            }
            for (component_name, cells) in &group.ports {
                let Some(component) = workflow.components.get(component_name) else {
                    errors.push(format!(
                        "KV service group '{group_name}' binds unknown component '{component_name}'"
                    ));
                    continue;
                };
                let inferred_ports = component.ports.inputs.is_empty()
                    && component.ports.outputs.is_empty()
                    && matches!(
                        component.implementation,
                        crate::schema::ComponentImplementation::Onnx { .. }
                    );
                for (cell_name, alias) in cells {
                    match workflow.state.get(cell_name) {
                        Some(state)
                            if state.service_group.as_deref() == Some(group_name.as_str()) => {}
                        Some(_) => errors.push(format!(
                            "KV service group '{group_name}' port alias references state \
                             '{cell_name}' bound to another service group"
                        )),
                        None => errors.push(format!(
                            "KV service group '{group_name}' port alias references unknown state \
                             '{cell_name}'"
                        )),
                    }
                    if !inferred_ports && !component.ports.inputs.contains_key(&alias.input) {
                        errors.push(format!(
                            "KV service group '{group_name}' component '{component_name}' input \
                             alias '{}' is not a declared port",
                            alias.input
                        ));
                    }
                    if !inferred_ports && !component.ports.outputs.contains_key(&alias.output) {
                        errors.push(format!(
                            "KV service group '{group_name}' component '{component_name}' output \
                             alias '{}' is not a declared port",
                            alias.output
                        ));
                    }
                }
            }
        }
    }

    let mut values = workflow.inputs.keys().cloned().collect::<BTreeSet<_>>();
    let mut value_contracts = workflow
        .inputs
        .iter()
        .map(|(name, input)| (name.clone(), input.contract.clone()))
        .collect::<BTreeMap<_, _>>();
    for input in workflow.inputs.values() {
        if let Some(present_as) = &input.present_as {
            values.insert(present_as.clone());
            value_contracts.insert(
                present_as.clone(),
                crate::schema::TensorContract {
                    dtype: "bool".to_string(),
                    rank: 0,
                    shape: Some(Vec::new()),
                    optional: false,
                },
            );
        }
    }
    let mut effects = compiled.initial_effects.clone();
    let mut effect_tokens = effects.values().cloned().collect::<BTreeSet<_>>();
    validate_workflow_node(
        &compiled.graph,
        workflow,
        &mut values,
        &mut value_contracts,
        &mut effects,
        &mut effect_tokens,
        "pipeline.workflow.steps",
        errors,
    );
    validate_emit_identity_consistency(&compiled.graph, &mut BTreeMap::new(), errors);
    if let Some(serving) = &workflow.serving {
        if serving.kv_service.compaction {
            validate_compacted_emit_identity(
                &compiled.graph,
                &serving.slot_ids,
                "pipeline.workflow.steps",
                false,
                errors,
            );
        }
        if serving.kv_service.groups.is_empty() {
            errors.push(
                "pipeline.workflow.serving.kv_service.groups must declare at least one bound \
                 state group"
                    .to_string(),
            );
        }
        if !serving.kv_service.groups.is_empty() && serving.accepted_len.is_none() {
            errors.push(
                "pipeline.workflow.serving.accepted_len is required when KV service groups are \
                 declared"
                    .to_string(),
            );
        }
        for (role, value) in [
            ("active", Some(&serving.active)),
            ("done", Some(&serving.done)),
            ("accepted_len", serving.accepted_len.as_ref()),
            ("slot_ids", Some(&serving.slot_ids)),
        ] {
            let Some(value) = value else {
                continue;
            };
            require_workflow_value(
                value,
                &values,
                &format!("pipeline.workflow.serving.{role}"),
                errors,
            );
            if let Some(contract) = value_contracts.get(value) {
                if matches!(role, "active" | "done") {
                    validate_bool_control_contract(
                        contract,
                        &format!("pipeline.workflow.serving.{role}"),
                        errors,
                    );
                } else {
                    validate_integer_control_contract(
                        contract,
                        &format!("pipeline.workflow.serving.{role}"),
                        errors,
                    );
                }
            }
        }
    }

    let mut used = BTreeSet::from(["workflow_ssa".to_string()]);
    if workflow.serving.is_some() {
        used.insert("serving_service_contract".to_string());
    }
    if workflow
        .state
        .values()
        .any(|state| state.scope == crate::schema::WorkflowStateScope::Session)
    {
        used.insert("session_state_lease".to_string());
    }
    if workflow.state.values().any(|state| {
        matches!(
            state.recurrence,
            crate::schema::ShapeRecurrence::Bounded { .. }
        )
    }) {
        used.insert("bounded_state_recurrence".to_string());
    }
    if workflow
        .state
        .values()
        .any(|state| state.class == crate::schema::WorkflowStateClass::Advisory)
    {
        used.insert("advisory_state".to_string());
    }
    if workflow
        .inputs
        .values()
        .any(|input| input.present_as.is_some())
    {
        used.insert("input_presence".to_string());
    }
    collect_workflow_capabilities(&compiled.graph, &mut used);
    for capability in used.difference(&workflow.manifest.capabilities) {
        errors.push(format!(
            "pipeline.workflow.manifest.capabilities is missing used capability '{capability}'"
        ));
    }
}

fn workflow_state_results(node: &WorkflowNode) -> BTreeMap<String, String> {
    match node {
        WorkflowNode::Sequence { nodes } => {
            let mut results = BTreeMap::new();
            for node in nodes {
                results.extend(workflow_state_results(node));
            }
            results
        }
        WorkflowNode::Loop { setup, carried, .. } => {
            let mut results = workflow_state_results(setup);
            for carry in carried {
                results.insert(carry.cell.clone(), carry.next.clone());
            }
            results
        }
        WorkflowNode::Branch {
            cases,
            default,
            outputs,
            ..
        } => {
            let case_results = cases
                .iter()
                .map(|(case, node)| (case, workflow_state_results(node)))
                .collect::<BTreeMap<_, _>>();
            let default_results = default.as_deref().map(workflow_state_results);
            let mut results: BTreeMap<String, String> = BTreeMap::new();
            for (output, phi) in outputs {
                let mut cell: Option<String> = None;
                for (case, source) in &phi.cases {
                    let Some((candidate, _)) = case_results
                        .get(case)
                        .and_then(|results| results.iter().find(|(_, value)| *value == source))
                    else {
                        continue;
                    };
                    match &cell {
                        Some(expected) if expected != candidate => {
                            cell = None;
                            break;
                        }
                        None => cell = Some((*candidate).clone()),
                        _ => {}
                    }
                }
                if let (Some(source), Some(default_results)) = (&phi.default, &default_results)
                    && let Some((candidate, _)) =
                        default_results.iter().find(|(_, value)| *value == source)
                {
                    match &cell {
                        Some(expected) if expected != candidate => {
                            cell = None;
                        }
                        None => cell = Some(candidate.clone()),
                        _ => {}
                    }
                }
                if let Some(cell) = cell {
                    results.insert(cell, output.clone());
                }
            }
            results
        }
        _ => BTreeMap::new(),
    }
}

// Recursive validation threads each independent symbol/effect table explicitly.
#[allow(clippy::too_many_arguments)]
fn validate_emit_identity_consistency(
    node: &WorkflowNode,
    outputs: &mut BTreeMap<String, bool>,
    errors: &mut Vec<String>,
) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for node in nodes {
                validate_emit_identity_consistency(node, outputs, errors);
            }
        }
        WorkflowNode::Loop { setup, body, .. } => {
            validate_emit_identity_consistency(setup, outputs, errors);
            validate_emit_identity_consistency(body, outputs, errors);
        }
        WorkflowNode::Branch { cases, default, .. } => {
            for case in cases.values() {
                validate_emit_identity_consistency(case, outputs, errors);
            }
            if let Some(default) = default {
                validate_emit_identity_consistency(default, outputs, errors);
            }
        }
        WorkflowNode::Emit {
            output, row_ids, ..
        } => {
            let row_wise = row_ids.is_some();
            if let Some(previous) = outputs.insert(output.clone(), row_wise)
                && previous != row_wise
            {
                errors.push(format!(
                    "pipeline.workflow output '{output}' mixes aggregate and row-wise emits; \
                     every emit for one output must agree on row_ids"
                ));
            }
        }
        WorkflowNode::Invoke { .. }
        | WorkflowNode::Transfer { .. }
        | WorkflowNode::ExecutionIsland { .. } => {}
    }
}

fn validate_compacted_emit_identity(
    node: &WorkflowNode,
    slot_ids: &str,
    path: &str,
    inside_loop: bool,
    errors: &mut Vec<String>,
) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for (index, node) in nodes.iter().enumerate() {
                validate_compacted_emit_identity(
                    node,
                    slot_ids,
                    &format!("{path}[{index}]"),
                    inside_loop,
                    errors,
                );
            }
        }
        WorkflowNode::Loop {
            setup,
            body,
            carried,
            ..
        } => {
            // Compaction happens between workflow runs, so the outer lifecycle loop must retain
            // semantic row identity. Nested control loops operate within that fixed permutation.
            if !inside_loop
                && !carried
                    .iter()
                    .any(|carry| carry.cell == slot_ids && carry.body_output == slot_ids)
            {
                errors.push(format!(
                    "{path}.carried must preserve serving slot_ids '{slot_ids}' when compaction \
                     is enabled; its next value must be the carried slot_ids value"
                ));
            }
            validate_compacted_emit_identity(
                setup,
                slot_ids,
                &format!("{path}.setup"),
                true,
                errors,
            );
            validate_compacted_emit_identity(body, slot_ids, &format!("{path}.body"), true, errors);
        }
        WorkflowNode::Branch { cases, default, .. } => {
            for (case, node) in cases {
                validate_compacted_emit_identity(
                    node,
                    slot_ids,
                    &format!("{path}.cases.{case}"),
                    inside_loop,
                    errors,
                );
            }
            if let Some(default) = default {
                validate_compacted_emit_identity(
                    default,
                    slot_ids,
                    &format!("{path}.default"),
                    inside_loop,
                    errors,
                );
            }
        }
        WorkflowNode::Emit {
            row_ids: Some(row_ids),
            ..
        } if row_ids != slot_ids => {
            errors.push(format!(
                "{path}.row_ids must reference serving slot_ids '{slot_ids}' when compaction is \
                 enabled"
            ));
        }
        WorkflowNode::Emit { .. }
        | WorkflowNode::Invoke { .. }
        | WorkflowNode::Transfer { .. }
        | WorkflowNode::ExecutionIsland { .. } => {}
    }
}

fn validate_workflow_node(
    node: &WorkflowNode,
    workflow: &WorkflowSpec,
    values: &mut BTreeSet<String>,
    value_contracts: &mut BTreeMap<String, crate::schema::TensorContract>,
    effects: &mut BTreeMap<String, String>,
    effect_tokens: &mut BTreeSet<String>,
    path: &str,
    errors: &mut Vec<String>,
) {
    match node {
        WorkflowNode::Sequence { nodes } => {
            if nodes.is_empty() && !path.ends_with(".setup") {
                errors.push(format!("{path}.nodes must not be empty"));
            }
            for (index, node) in nodes.iter().enumerate() {
                validate_workflow_node(
                    node,
                    workflow,
                    values,
                    value_contracts,
                    effects,
                    effect_tokens,
                    &format!("{path}.nodes[{index}]"),
                    errors,
                );
            }
        }
        WorkflowNode::Invoke {
            component,
            inputs,
            outputs,
            effects: transitions,
        } => {
            let Some(declaration) = workflow.components.get(component) else {
                errors.push(format!("{path} invokes unknown component '{component}'"));
                return;
            };
            let inferred_onnx_ports = declaration.ports.inputs.is_empty()
                && declaration.ports.outputs.is_empty()
                && matches!(
                    declaration.implementation,
                    crate::schema::ComponentImplementation::Onnx { .. }
                );
            for (port, value) in inputs {
                if !inferred_onnx_ports && !declaration.ports.inputs.contains_key(port) {
                    errors.push(format!("{path}.inputs has unknown port '{port}'"));
                }
                require_workflow_value(value, values, &format!("{path}.inputs.{port}"), errors);
                if let (Some(source), Some(target)) = (
                    value_contracts.get(value),
                    declaration.ports.inputs.get(port),
                ) {
                    require_compatible_contracts(
                        source,
                        target,
                        &format!("{path}.inputs.{port}"),
                        errors,
                    );
                }
            }
            if !inferred_onnx_ports {
                for (port, contract) in &declaration.ports.inputs {
                    if !contract.optional && !inputs.contains_key(port) {
                        errors.push(format!("{path}.inputs is missing port '{port}'"));
                    }
                }
            }
            for (port, value) in outputs {
                if !inferred_onnx_ports && !declaration.ports.outputs.contains_key(port) {
                    errors.push(format!("{path}.outputs has unknown port '{port}'"));
                }
                define_workflow_value(value, values, &format!("{path}.outputs.{port}"), errors);
                if let Some(contract) = declaration.ports.outputs.get(port) {
                    value_contracts.insert(value.clone(), contract.clone());
                }
            }
            if !inferred_onnx_ports {
                for (port, contract) in &declaration.ports.outputs {
                    if !contract.optional && !outputs.contains_key(port) {
                        errors.push(format!("{path}.outputs is missing port '{port}'"));
                    }
                }
            }
            for effect in &declaration.effects {
                if !transitions.contains_key(effect) {
                    errors.push(format!(
                        "{path}.effects is missing declared effect '{effect}'"
                    ));
                }
            }
            for (effect, transition) in transitions {
                apply_effect_transition(
                    effect,
                    transition,
                    effects,
                    effect_tokens,
                    &format!("{path}.effects.{effect}"),
                    errors,
                );
            }
        }
        WorkflowNode::Loop {
            setup,
            body,
            continue_when,
            max_iterations,
            iteration,
            carried,
            effects: loop_effects,
            termination: _,
        } => {
            validate_workflow_node(
                setup,
                workflow,
                values,
                value_contracts,
                effects,
                effect_tokens,
                &format!("{path}.setup"),
                errors,
            );
            require_workflow_value(
                max_iterations,
                values,
                &format!("{path}.max_iterations"),
                errors,
            );
            if let Some(contract) = value_contracts.get(max_iterations) {
                validate_integer_scalar_contract(
                    contract,
                    &format!("{path}.max_iterations"),
                    errors,
                );
            }
            let mut body_values = values.clone();
            let mut body_contracts = value_contracts.clone();
            let mut body_effects = effects.clone();
            for (domain, merge) in loop_effects {
                match effects.get(domain) {
                    Some(incoming) if incoming == &merge.incoming => {}
                    Some(incoming) => errors.push(format!(
                        "{path}.effects.{domain} consumes '{}', but the incoming token is \
                         '{incoming}'",
                        merge.incoming
                    )),
                    None => errors.push(format!(
                        "{path}.effects.{domain} references undeclared effect"
                    )),
                }
                if !effect_tokens.insert(merge.body_input.clone()) {
                    errors.push(format!(
                        "{path}.effects.{domain}.body_input '{}' is already produced",
                        merge.body_input
                    ));
                }
                body_effects.insert(domain.clone(), merge.body_input.clone());
            }
            if let Some(iteration) = iteration {
                if iteration.contract.dtype != "int64" || !matches!(iteration.contract.rank, 0 | 1)
                {
                    errors.push(format!(
                        "{path}.iteration must declare int64 rank 0 or rank 1, got {} rank {}",
                        iteration.contract.dtype, iteration.contract.rank
                    ));
                }
                match (iteration.contract.rank, iteration.contract.shape.as_ref()) {
                    (0, Some(shape)) if !shape.is_empty() => errors.push(format!(
                        "{path}.iteration scalar contract must have an empty shape"
                    )),
                    (1, Some(shape)) if shape.len() != 1 => errors.push(format!(
                        "{path}.iteration rank-one broadcast contract must have one dimension"
                    )),
                    (1, None) => errors.push(format!(
                        "{path}.iteration rank-one broadcast contract must declare its shape"
                    )),
                    _ => {}
                }
                define_workflow_value(
                    &iteration.value,
                    &mut body_values,
                    &format!("{path}.iteration"),
                    errors,
                );
                body_contracts.insert(iteration.value.clone(), iteration.contract.clone());
            }
            for carry in carried {
                let Some(state) = workflow.state.get(&carry.cell) else {
                    errors.push(format!(
                        "{path}.carried references unknown cell '{}'",
                        carry.cell
                    ));
                    continue;
                };
                let recurrence_values = match &state.recurrence {
                    crate::schema::ShapeRecurrence::Invariant => Vec::new(),
                    crate::schema::ShapeRecurrence::Growing { increment, max, .. } => {
                        vec![("increment", increment), ("max", max)]
                    }
                    crate::schema::ShapeRecurrence::Bounded { max, .. } => {
                        vec![("max", max)]
                    }
                };
                for (name, value) in recurrence_values {
                    let recurrence_path =
                        format!("{path}.carried.{}.recurrence.{name}", carry.cell);
                    require_workflow_value(value, values, &recurrence_path, errors);
                    if let Some(contract) = value_contracts.get(value) {
                        validate_integer_scalar_contract(contract, &recurrence_path, errors);
                    }
                }
                require_workflow_value(
                    &state.initializer,
                    values,
                    &format!("{path}.carried.initializer"),
                    errors,
                );
                if let Some(initializer_contract) = value_contracts.get(&state.initializer) {
                    require_state_contract(
                        initializer_contract,
                        &state.contract,
                        &state.recurrence,
                        false,
                        &format!("{path}.carried.initializer"),
                        errors,
                    );
                }
                require_workflow_value(
                    &carry.current,
                    values,
                    &format!("{path}.carried.current"),
                    errors,
                );
                if let Some(current_contract) = value_contracts.get(&carry.current) {
                    require_state_contract(
                        current_contract,
                        &state.contract,
                        &state.recurrence,
                        false,
                        &format!("{path}.carried.current"),
                        errors,
                    );
                }
                body_values.insert(carry.body_input.clone());
                body_contracts.insert(carry.body_input.clone(), state.contract.clone());
                apply_effect_transition(
                    &format!("state:{}", carry.cell),
                    &carry.read_effect,
                    &mut body_effects,
                    effect_tokens,
                    &format!("{path}.carried.read_effect"),
                    errors,
                );
            }
            require_workflow_value(
                continue_when,
                &body_values,
                &format!("{path}.continue_when"),
                errors,
            );
            if let Some(contract) = body_contracts.get(continue_when) {
                validate_bool_control_contract(contract, &format!("{path}.continue_when"), errors);
            }
            validate_workflow_node(
                body,
                workflow,
                &mut body_values,
                &mut body_contracts,
                &mut body_effects,
                effect_tokens,
                &format!("{path}.body"),
                errors,
            );
            for carry in carried {
                require_workflow_value(
                    &carry.body_output,
                    &body_values,
                    &format!("{path}.carried.body_output"),
                    errors,
                );
                if let (Some(next_contract), Some(state)) = (
                    body_contracts.get(&carry.body_output),
                    workflow.state.get(&carry.cell),
                ) {
                    require_state_contract(
                        next_contract,
                        &state.contract,
                        &state.recurrence,
                        true,
                        &format!("{path}.carried.body_output"),
                        errors,
                    );
                }
                apply_effect_transition(
                    &format!("state:{}", carry.cell),
                    &carry.write_effect,
                    &mut body_effects,
                    effect_tokens,
                    &format!("{path}.carried.write_effect"),
                    errors,
                );
                define_workflow_value(&carry.next, values, &format!("{path}.carried.next"), errors);
                if let Some(contract) = body_contracts.get(&carry.body_output) {
                    value_contracts.insert(carry.next.clone(), contract.clone());
                }
            }
            for (domain, merge) in loop_effects {
                if body_effects.get(domain) != Some(&merge.body_output) {
                    errors.push(format!(
                        "{path}.effects.{domain} body produces '{}', but the final body token is \
                         {:?}",
                        merge.body_output,
                        body_effects.get(domain)
                    ));
                }
                if !effect_tokens.insert(merge.produces.clone()) {
                    errors.push(format!(
                        "{path}.effects.{domain}.produces '{}' is already produced",
                        merge.produces
                    ));
                }
                effects.insert(domain.clone(), merge.produces.clone());
            }
        }
        WorkflowNode::Branch {
            predicate,
            cases,
            default,
            outputs,
            effects: merges,
        } => {
            require_workflow_value(predicate, values, &format!("{path}.predicate"), errors);
            if let Some(contract) = value_contracts.get(predicate) {
                validate_predicate_contract(contract, &format!("{path}.predicate"), errors);
            }
            if cases.is_empty() {
                errors.push(format!("{path}.cases must not be empty"));
            }
            let mut case_scopes = BTreeMap::new();
            for (case, node) in cases {
                let mut case_values = values.clone();
                let mut case_contracts = value_contracts.clone();
                let mut case_effects = effects.clone();
                let mut case_tokens = effect_tokens.clone();
                validate_workflow_node(
                    node,
                    workflow,
                    &mut case_values,
                    &mut case_contracts,
                    &mut case_effects,
                    &mut case_tokens,
                    &format!("{path}.cases.{case}"),
                    errors,
                );
                case_scopes.insert(
                    case.clone(),
                    (case_values, case_contracts, case_effects, case_tokens),
                );
            }
            let default_scope = if let Some(default) = default {
                let mut default_values = values.clone();
                let mut default_contracts = value_contracts.clone();
                let mut default_effects = effects.clone();
                let mut default_tokens = effect_tokens.clone();
                validate_workflow_node(
                    default,
                    workflow,
                    &mut default_values,
                    &mut default_contracts,
                    &mut default_effects,
                    &mut default_tokens,
                    &format!("{path}.default"),
                    errors,
                );
                Some((
                    default_values,
                    default_contracts,
                    default_effects,
                    default_tokens,
                ))
            } else {
                None
            };

            for (case, node) in cases {
                for (cell, source) in workflow_state_results(node) {
                    if workflow.state.get(&cell).is_some_and(|state| {
                        state.scope == crate::schema::WorkflowStateScope::Session
                    }) && !outputs
                        .values()
                        .any(|phi| phi.cases.get(case) == Some(&source))
                    {
                        errors.push(format!(
                            "{path}.cases.{case} updates session state '{cell}' to '{source}' \
                             without exporting it through a branch output"
                        ));
                    }
                }
            }
            if let Some(default) = default {
                for (cell, source) in workflow_state_results(default) {
                    if workflow.state.get(&cell).is_some_and(|state| {
                        state.scope == crate::schema::WorkflowStateScope::Session
                    }) && !outputs
                        .values()
                        .any(|phi| phi.default.as_ref() == Some(&source))
                    {
                        errors.push(format!(
                            "{path}.default updates session state '{cell}' to '{source}' \
                             without exporting it through a branch output"
                        ));
                    }
                }
            }

            for (output, phi) in outputs {
                for case in cases.keys() {
                    if !phi.cases.contains_key(case) {
                        errors.push(format!(
                            "{path}.outputs.{output} has no value for case '{case}'"
                        ));
                    }
                }
                for case in phi.cases.keys() {
                    if !cases.contains_key(case) {
                        errors.push(format!(
                            "{path}.outputs.{output} references unknown case '{case}'"
                        ));
                    }
                }
                if default.is_some() != phi.default.is_some() {
                    errors.push(format!(
                        "{path}.outputs.{output} must map the default branch exactly when one \
                         is declared"
                    ));
                }

                let mut contract = None;
                for (case, source) in &phi.cases {
                    let Some((case_values, case_contracts, _, _)) = case_scopes.get(case) else {
                        continue;
                    };
                    require_workflow_value(
                        source,
                        case_values,
                        &format!("{path}.outputs.{output}.cases.{case}"),
                        errors,
                    );
                    if let Some(source_contract) = case_contracts.get(source) {
                        if let Some(expected) = &contract {
                            require_compatible_contracts(
                                source_contract,
                                expected,
                                &format!("{path}.outputs.{output}.cases.{case}"),
                                errors,
                            );
                        } else {
                            contract = Some(source_contract.clone());
                        }
                    }
                }
                if let (Some(source), Some((default_values, default_contracts, _, _))) =
                    (&phi.default, &default_scope)
                {
                    require_workflow_value(
                        source,
                        default_values,
                        &format!("{path}.outputs.{output}.default"),
                        errors,
                    );
                    if let Some(source_contract) = default_contracts.get(source) {
                        if let Some(expected) = &contract {
                            require_compatible_contracts(
                                source_contract,
                                expected,
                                &format!("{path}.outputs.{output}.default"),
                                errors,
                            );
                        } else {
                            contract = Some(source_contract.clone());
                        }
                    }
                }
                define_workflow_value(output, values, &format!("{path}.outputs.{output}"), errors);
                if let Some(contract) = contract {
                    value_contracts.insert(output.clone(), contract);
                }
            }

            for (effect_name, merge) in merges {
                match effects.get(effect_name) {
                    Some(incoming) if incoming == &merge.incoming => {}
                    Some(incoming) => errors.push(format!(
                        "{path}.effects.{effect_name} consumes '{}', but the incoming token is \
                         '{incoming}'",
                        merge.incoming
                    )),
                    None => errors.push(format!(
                        "{path}.effects.{effect_name} references undeclared effect"
                    )),
                }
                for case in cases.keys() {
                    if !merge.cases.contains_key(case) {
                        errors.push(format!(
                            "{path}.effects.{effect_name} has no successor for case '{case}'"
                        ));
                    }
                }
                for case in merge.cases.keys() {
                    if !cases.contains_key(case) {
                        errors.push(format!(
                            "{path}.effects.{effect_name} references unknown case '{case}'"
                        ));
                    }
                }
                if default.is_some() != merge.default.is_some() {
                    errors.push(format!(
                        "{path}.effects.{effect_name} must map the default branch exactly when \
                         one is declared"
                    ));
                }
                for (case, successor) in &merge.cases {
                    if let Some((_, _, case_effects, _)) = case_scopes.get(case)
                        && case_effects.get(effect_name) != Some(successor)
                    {
                        errors.push(format!(
                            "{path}.effects.{effect_name}.cases.{case} declares successor \
                             '{successor}', but the case produces {:?}",
                            case_effects.get(effect_name)
                        ));
                    }
                }
                if let (Some(successor), Some((_, _, default_effects, _))) =
                    (&merge.default, &default_scope)
                    && default_effects.get(effect_name) != Some(successor)
                {
                    errors.push(format!(
                        "{path}.effects.{effect_name}.default declares successor \
                         '{successor}', but the default produces {:?}",
                        default_effects.get(effect_name)
                    ));
                }
                if merge.cases.values().any(|token| token == &merge.produces)
                    || merge.default.as_ref() == Some(&merge.produces)
                    || case_scopes
                        .values()
                        .any(|(_, _, _, tokens)| tokens.contains(&merge.produces))
                    || default_scope
                        .as_ref()
                        .is_some_and(|(_, _, _, tokens)| tokens.contains(&merge.produces))
                {
                    errors.push(format!(
                        "{path}.effects.{effect_name} joined successor '{}' must be distinct \
                         from every case-local successor",
                        merge.produces
                    ));
                }
                if !effect_tokens.insert(merge.produces.clone()) {
                    errors.push(format!(
                        "{path}.effects.{effect_name} produces duplicate effect token '{}'",
                        merge.produces
                    ));
                }
                effects.insert(effect_name.clone(), merge.produces.clone());
            }

            for (case, (_, _, case_effects, _)) in &case_scopes {
                for (effect_name, successor) in case_effects {
                    if effects.get(effect_name) != Some(successor)
                        && !merges.contains_key(effect_name)
                    {
                        errors.push(format!(
                            "{path}.cases.{case} changes effect '{effect_name}' without an \
                             explicit branch merge"
                        ));
                    }
                }
            }
            if let Some((_, _, default_effects, _)) = &default_scope {
                for (effect_name, successor) in default_effects {
                    if effects.get(effect_name) != Some(successor)
                        && !merges.contains_key(effect_name)
                    {
                        errors.push(format!(
                            "{path}.default changes effect '{effect_name}' without an explicit \
                             branch merge"
                        ));
                    }
                }
            }
        }
        WorkflowNode::Emit {
            value,
            when,
            valid_length,
            row_ids,
            output,
            effect_name,
            effect,
            ..
        } => {
            require_workflow_value(value, values, &format!("{path}.value"), errors);
            if let Some(when) = when {
                require_workflow_value(when, values, &format!("{path}.when"), errors);
                if let Some(contract) = value_contracts.get(when) {
                    validate_bool_control_contract(contract, &format!("{path}.when"), errors);
                }
            }
            if let Some(valid_length) = valid_length {
                require_workflow_value(
                    valid_length,
                    values,
                    &format!("{path}.valid_length"),
                    errors,
                );
                if let Some(contract) = value_contracts.get(valid_length) {
                    validate_integer_control_contract(
                        contract,
                        &format!("{path}.valid_length"),
                        errors,
                    );
                }
            }
            if let Some(row_ids) = row_ids {
                require_workflow_value(row_ids, values, &format!("{path}.row_ids"), errors);
                if let Some(contract) = value_contracts.get(row_ids) {
                    if contract.dtype != "int64" || contract.rank != 1 {
                        errors.push(format!("{path}.row_ids must be int64[B]"));
                    }
                    if let (Some(row_batch), Some(value_batch)) = (
                        contract.shape.as_deref().and_then(|shape| shape.first()),
                        value_contracts
                            .get(value)
                            .and_then(|contract| contract.shape.as_deref())
                            .and_then(|shape| shape.first()),
                    ) && row_batch != value_batch
                    {
                        errors.push(format!(
                            "{path}.row_ids batch dimension must match {path}.value"
                        ));
                    }
                }
            } else {
                for (field, control) in [
                    ("when", when.as_ref()),
                    ("valid_length", valid_length.as_ref()),
                ] {
                    let Some(contract) = control.and_then(|name| value_contracts.get(name)) else {
                        continue;
                    };
                    let singleton = contract.rank == 0
                        || (contract.rank == 1
                            && matches!(
                                contract.shape.as_deref().and_then(|shape| shape.first()),
                                Some(crate::schema::TensorDimension::Fixed(1))
                            ));
                    if !singleton {
                        errors.push(format!(
                            "{path}.{field} is row-wise and requires explicit row_ids"
                        ));
                    }
                }
            }
            if !workflow.outputs.contains_key(output) {
                errors.push(format!("{path} emits undeclared output '{output}'"));
            } else if let (Some(value_contract), Some(output_contract)) = (
                value_contracts.get(value),
                workflow.outputs.get(output).map(|output| &output.contract),
            ) {
                if valid_length.is_some() {
                    require_emit_prefix_contracts(
                        value_contract,
                        output_contract,
                        &format!("{path}.output"),
                        errors,
                    );
                } else {
                    require_compatible_contracts(
                        value_contract,
                        output_contract,
                        &format!("{path}.output"),
                        errors,
                    );
                }
            }
            apply_effect_transition(
                effect_name,
                effect,
                effects,
                effect_tokens,
                &format!("{path}.effect"),
                errors,
            );
        }
        WorkflowNode::Transfer { input, output, .. } => {
            require_workflow_value(input, values, &format!("{path}.input"), errors);
            define_workflow_value(output, values, &format!("{path}.output"), errors);
            if let Some(contract) = value_contracts.get(input).cloned() {
                value_contracts.insert(output.clone(), contract);
            }
        }
        WorkflowNode::ExecutionIsland { .. } => {}
    }
}

fn validate_integer_scalar_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if !matches!(
        contract.dtype.as_str(),
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
    ) {
        errors.push(format!("{path} must have an integer dtype"));
    }
    match contract.rank {
        0 => {
            if contract
                .shape
                .as_ref()
                .is_some_and(|shape| !shape.is_empty())
            {
                errors.push(format!("{path} rank-zero contract must have shape []"));
            }
        }
        1 => {
            if !matches!(
                contract.shape.as_deref(),
                Some([crate::schema::TensorDimension::Fixed(1)])
            ) {
                errors.push(format!(
                    "{path} rank-one control contract must have static shape [1]"
                ));
            }
        }
        _ => errors.push(format!("{path} must be a scalar or rank-one tensor")),
    }
}

fn validate_integer_control_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if !matches!(
        contract.dtype.as_str(),
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
    ) {
        errors.push(format!("{path} must have an integer dtype"));
    }
    if !matches!(contract.rank, 0 | 1) {
        errors.push(format!("{path} must be a scalar or rank-one tensor"));
    }
}

fn validate_bool_control_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if contract.dtype != "bool" {
        errors.push(format!("{path} must have bool dtype"));
    }
    if !matches!(contract.rank, 0 | 1) {
        errors.push(format!("{path} must be a scalar or rank-one row tensor"));
    }
}

fn validate_predicate_contract(
    contract: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if contract.dtype != "bool"
        && !matches!(
            contract.dtype.as_str(),
            "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
        )
    {
        errors.push(format!("{path} must have a bool or integer dtype"));
    }
    if !matches!(contract.rank, 0 | 1) {
        errors.push(format!(
            "{path} must be a scalar or rank-one broadcast tensor"
        ));
    } else if contract.rank == 1
        && !matches!(
            contract.shape.as_deref(),
            Some([crate::schema::TensorDimension::Fixed(1)])
        )
    {
        errors.push(format!(
            "{path} rank-one predicate contract must have static shape [1]"
        ));
    }
}

fn validate_optional_input_guards(
    node: &WorkflowNode,
    input: &str,
    present_as: &str,
    guaranteed_present: bool,
    path: &str,
    errors: &mut Vec<String>,
) {
    let check = |value: &str, value_path: &str, errors: &mut Vec<String>| {
        if value == input && !guaranteed_present {
            errors.push(format!(
                "{value_path} reads optional input '{input}' outside the true case of its \
                     present_as predicate '{present_as}'"
            ));
        }
    };
    match node {
        WorkflowNode::Sequence { nodes } => {
            for (index, node) in nodes.iter().enumerate() {
                validate_optional_input_guards(
                    node,
                    input,
                    present_as,
                    guaranteed_present,
                    &format!("{path}.nodes[{index}]"),
                    errors,
                );
            }
        }
        WorkflowNode::Invoke {
            inputs: invoke_inputs,
            ..
        } => {
            for (port, value) in invoke_inputs {
                check(value, &format!("{path}.inputs.{port}"), errors);
            }
        }
        WorkflowNode::Loop {
            setup,
            body,
            continue_when,
            max_iterations,
            carried,
            ..
        } => {
            check(continue_when, &format!("{path}.continue_when"), errors);
            check(max_iterations, &format!("{path}.max_iterations"), errors);
            for (index, carry) in carried.iter().enumerate() {
                check(
                    &carry.current,
                    &format!("{path}.carried[{index}].current"),
                    errors,
                );
                check(
                    &carry.body_output,
                    &format!("{path}.carried[{index}].body_output"),
                    errors,
                );
            }
            validate_optional_input_guards(
                setup,
                input,
                present_as,
                guaranteed_present,
                &format!("{path}.setup"),
                errors,
            );
            validate_optional_input_guards(
                body,
                input,
                present_as,
                guaranteed_present,
                &format!("{path}.body"),
                errors,
            );
        }
        WorkflowNode::Branch {
            predicate,
            cases,
            default,
            outputs,
            ..
        } => {
            check(predicate, &format!("{path}.predicate"), errors);
            for (case, node) in cases {
                let case_present =
                    guaranteed_present || (predicate == present_as && case == "true");
                validate_optional_input_guards(
                    node,
                    input,
                    present_as,
                    case_present,
                    &format!("{path}.cases.{case}"),
                    errors,
                );
            }
            if let Some(default) = default {
                validate_optional_input_guards(
                    default,
                    input,
                    present_as,
                    guaranteed_present,
                    &format!("{path}.default"),
                    errors,
                );
            }
            for (output, phi) in outputs {
                for (case, value) in &phi.cases {
                    let case_present =
                        guaranteed_present || (predicate == present_as && case == "true");
                    if value == input && !case_present {
                        errors.push(format!(
                            "{path}.outputs.{output}.cases.{case} reads optional input '{input}' \
                                 outside its presence branch"
                        ));
                    }
                }
                if let Some(value) = &phi.default {
                    check(value, &format!("{path}.outputs.{output}.default"), errors);
                }
            }
        }
        WorkflowNode::Emit {
            value,
            when,
            valid_length,
            row_ids,
            ..
        } => {
            check(value, &format!("{path}.value"), errors);
            if let Some(when) = when {
                check(when, &format!("{path}.when"), errors);
            }
            if let Some(valid_length) = valid_length {
                check(valid_length, &format!("{path}.valid_length"), errors);
            }
            if let Some(row_ids) = row_ids {
                check(row_ids, &format!("{path}.row_ids"), errors);
            }
        }
        WorkflowNode::Transfer {
            input: transfer_input,
            ..
        } => check(transfer_input, &format!("{path}.input"), errors),
        WorkflowNode::ExecutionIsland { .. } => {}
    }
}

fn require_emit_prefix_contracts(
    actual: &crate::schema::TensorContract,
    declared: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    if actual.rank == 0 || declared.rank == 0 {
        errors.push(format!(
            "{path} valid_length requires emitted value and output contracts with rank >= 1"
        ));
        return;
    }
    if actual.dtype != declared.dtype || actual.rank != declared.rank {
        errors.push(format!(
            "{path} has incompatible dtype or rank for prefix emission"
        ));
        return;
    }
    let (Some(actual_shape), Some(declared_shape)) = (&actual.shape, &declared.shape) else {
        return;
    };
    let prefix_axis = actual.rank.saturating_sub(1);
    for (axis, (actual, declared)) in actual_shape.iter().zip(declared_shape).enumerate() {
        if axis != prefix_axis
            && matches!(
                (actual, declared),
                (
                    crate::schema::TensorDimension::Fixed(_),
                    crate::schema::TensorDimension::Fixed(_)
                )
            )
            && actual != declared
        {
            errors.push(format!(
                "{path} has incompatible fixed dimension at axis {axis}"
            ));
        }
    }
}

fn require_workflow_value(
    value: &str,
    values: &BTreeSet<String>,
    path: &str,
    errors: &mut Vec<String>,
) {
    if !values.contains(value) {
        errors.push(format!(
            "{path} references value '{value}' before definition"
        ));
    }
}

fn define_workflow_value(
    value: &str,
    values: &mut BTreeSet<String>,
    path: &str,
    errors: &mut Vec<String>,
) {
    if !values.insert(value.to_string()) {
        errors.push(format!("{path} redefines SSA value '{value}'"));
    }
}

fn require_compatible_contracts(
    source: &crate::schema::TensorContract,
    target: &crate::schema::TensorContract,
    path: &str,
    errors: &mut Vec<String>,
) {
    fn normalize_dtype(dtype: &str) -> &str {
        match dtype {
            "fp32" => "float32",
            "fp16" => "float16",
            "bf16" => "bfloat16",
            other => other,
        }
    }
    if normalize_dtype(&source.dtype) != normalize_dtype(&target.dtype)
        || source.rank != target.rank
    {
        errors.push(format!(
            "{path} has incompatible tensor contracts: {} rank {} -> {} rank {}",
            source.dtype, source.rank, target.dtype, target.rank
        ));
        return;
    }
    if let (Some(source_shape), Some(target_shape)) = (&source.shape, &target.shape) {
        for (axis, (source, target)) in source_shape.iter().zip(target_shape).enumerate() {
            if matches!(
                (source, target),
                (
                    crate::schema::TensorDimension::Fixed(_),
                    crate::schema::TensorDimension::Fixed(_)
                )
            ) && source != target
            {
                errors.push(format!(
                    "{path} has incompatible fixed dimension at axis {axis}"
                ));
            }
        }
    }
}

fn require_state_contract(
    actual: &crate::schema::TensorContract,
    declared: &crate::schema::TensorContract,
    recurrence: &crate::schema::ShapeRecurrence,
    next: bool,
    path: &str,
    errors: &mut Vec<String>,
) {
    fn normalize_dtype(dtype: &str) -> &str {
        match dtype {
            "fp32" => "float32",
            "fp16" => "float16",
            "bf16" => "bfloat16",
            other => other,
        }
    }
    if normalize_dtype(&actual.dtype) != normalize_dtype(&declared.dtype)
        || actual.rank != declared.rank
    {
        errors.push(format!(
            "{path} is incompatible with state contract {} rank {}",
            declared.dtype, declared.rank
        ));
        return;
    }
    let (Some(actual_shape), Some(declared_shape)) = (&actual.shape, &declared.shape) else {
        return;
    };
    let dynamic_axis = match recurrence {
        crate::schema::ShapeRecurrence::Growing { axis, .. } if next => Some(*axis),
        crate::schema::ShapeRecurrence::Bounded { axis, .. } => Some(*axis),
        _ => None,
    };
    for (axis, (actual, declared)) in actual_shape.iter().zip(declared_shape).enumerate() {
        if Some(axis) != dynamic_axis
            && matches!(
                (actual, declared),
                (
                    crate::schema::TensorDimension::Fixed(_),
                    crate::schema::TensorDimension::Fixed(_)
                )
            )
            && actual != declared
        {
            errors.push(format!(
                "{path} has incompatible state dimension at axis {axis}"
            ));
        }
    }
}

fn apply_effect_transition(
    effect: &str,
    transition: &crate::schema::EffectTransition,
    effects: &mut BTreeMap<String, String>,
    tokens: &mut BTreeSet<String>,
    path: &str,
    errors: &mut Vec<String>,
) {
    match effects.get(effect) {
        Some(current) if current == &transition.consumes => {}
        Some(current) => errors.push(format!(
            "{path} consumes effect token '{}', but current token is '{current}'",
            transition.consumes
        )),
        None => errors.push(format!("{path} references undeclared effect '{effect}'")),
    }
    if !tokens.insert(transition.produces.clone()) {
        errors.push(format!(
            "{path} produces duplicate effect token '{}'",
            transition.produces
        ));
    }
    effects.insert(effect.to_string(), transition.produces.clone());
}

/// Error type for metadata operations.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Unsupported capabilities: {0:?}")]
    Unsupported(Vec<String>),
}

// Re-export at crate level
pub use MetadataError as Error;
