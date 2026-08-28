use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use onnx_genai_metadata::{PipelineSpec, TensorDimension};
use onnx_runtime_loader::proto::onnx::{ValueInfoProto, tensor_shape_proto, type_proto};
use onnx_std::ir::{DataType, Dim};

use crate::{OrtError, Result};

#[derive(Debug, Clone)]
struct PortSignature {
    dtype: DataType,
    shape: Vec<PortDimension>,
}

#[derive(Debug, Clone, Copy)]
enum PortDimension {
    Static(usize),
    Dynamic,
}

impl PortSignature {
    fn rank(&self) -> usize {
        self.shape.len()
    }
}

#[derive(Debug, Default)]
struct ComponentSignature {
    inputs: BTreeMap<String, PortSignature>,
    outputs: BTreeMap<String, PortSignature>,
    defaulted_inputs: BTreeSet<String>,
    /// Names of every initializer (graph weight) the loaded ONNX model owns.
    ///
    /// Populated from the loader's decoded model inventory, so external-data
    /// initializers resolved from sidecar files are included. This is what a
    /// speculative shared-weight relationship must resolve against.
    initializers: BTreeSet<String>,
}

pub(crate) fn validate_pipeline_admission(
    spec: &PipelineSpec,
    model_paths: &BTreeMap<String, PathBuf>,
    speculative: Option<&onnx_genai_metadata::SpeculativeContract>,
) -> Result<()> {
    let signatures = inspect_component_signatures(model_paths)?;
    validate_workflow_signatures(&spec.workflow, &signatures)?;
    if let Some(speculative) = speculative {
        validate_speculative_shared_initializers(speculative, &signatures)?;
    }
    Ok(())
}

/// Resolve every declared speculative shared weight against the target model's
/// real initializer inventory, fail-closed.
///
/// The metadata validator already checks the *structural* contract: the
/// embedding component must be the speculative target, that target must be an
/// ONNX component, and the table string must be non-empty. It cannot, however,
/// see inside the ONNX artifact, so a producer can still name a table that does
/// not exist. A folded-carry proposer gathers `embed(last_token)` from this
/// exact initializer, while DFlash borrows both the target embedding and output
/// projection. A dangling relationship would silently break proposal semantics.
/// Admission has already inspected every ONNX component, so each declared name
/// must match a real target initializer.
fn validate_speculative_shared_initializers(
    speculative: &onnx_genai_metadata::SpeculativeContract,
    signatures: &BTreeMap<String, ComponentSignature>,
) -> Result<()> {
    let required = match &speculative.proposal_execution {
        onnx_genai_metadata::SpeculativeProposalExecution::Chained {
            token_embedding: Some(embedding),
            ..
        } => vec![(
            "token_embedding.table",
            embedding.component.as_str(),
            embedding.table.as_str(),
        )],
        onnx_genai_metadata::SpeculativeProposalExecution::DflashFlatBlock {
            shared_weights,
            ..
        } => vec![
            (
                "DFlash input_embedding.table",
                shared_weights.input_embedding.component.as_str(),
                shared_weights.input_embedding.table.as_str(),
            ),
            (
                "DFlash output_projection.initializer",
                shared_weights.output_projection.component.as_str(),
                shared_weights.output_projection.initializer.as_str(),
            ),
        ],
        _ => Vec::new(),
    };
    for (field, component, initializer) in required {
        let signature = signatures.get(component).ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "package admission rejected speculative {field}: component '{component}' has no \
                 inspected ONNX model, so initializer '{initializer}' cannot be resolved. How \
                 to fix: name the speculative target ONNX component that owns the initializer"
            ))
        })?;
        if !signature.initializers.contains(initializer) {
            return Err(OrtError::InvalidArgument(format!(
                "package admission rejected speculative {field}: '{initializer}' is not an \
                 initializer of target component '{component}'. Shared speculative weights are \
                 immutable target relationships, not names the runtime may guess or duplicate. \
                 How to fix: emit the exact target initializer name (declared initializers: \
                 {available})",
                available = summarize_initializers(&signature.initializers),
            )));
        }
    }
    Ok(())
}

/// Render an initializer inventory for an error message: sorted, capped, and
/// annotated with the total so a large target model does not dump thousands of
/// names while still pointing a producer at the real weights.
fn summarize_initializers(initializers: &BTreeSet<String>) -> String {
    const MAX_LISTED: usize = 8;
    if initializers.is_empty() {
        return "none".to_string();
    }
    let listed = initializers
        .iter()
        .take(MAX_LISTED)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if initializers.len() > MAX_LISTED {
        format!("{listed}, ... ({total} total)", total = initializers.len())
    } else {
        listed
    }
}

fn validate_workflow_signatures(
    workflow: &onnx_genai_metadata::WorkflowSpec,
    signatures: &BTreeMap<String, ComponentSignature>,
) -> Result<()> {
    for (component, declaration) in &workflow.components {
        let onnx_genai_metadata::ComponentImplementation::Onnx { .. } = &declaration.implementation
        else {
            continue;
        };
        let signature = signatures.get(component).ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "workflow ONNX component '{component}' has no inspected model"
            ))
        })?;
        for (direction, declared, actual) in [
            ("input", &declaration.ports.inputs, &signature.inputs),
            ("output", &declaration.ports.outputs, &signature.outputs),
        ] {
            for (port, contract) in declared {
                let signature = actual.get(port).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "workflow component '{component}' declares {direction} port '{port}', \
                         but the ONNX graph does not expose it"
                    ))
                })?;
                let dtype = parse_dtype(&contract.dtype).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "workflow component '{component}' {direction} '{port}' has unsupported \
                         dtype '{}'",
                        contract.dtype
                    ))
                })?;
                if dtype != signature.dtype {
                    return Err(OrtError::InvalidArgument(format!(
                        "workflow component '{component}' {direction} '{port}' declares dtype {}, \
                         but the ONNX graph exposes {}",
                        contract.dtype,
                        dtype_name(signature.dtype)
                    )));
                }
                if contract.rank() != signature.rank() {
                    return Err(OrtError::InvalidArgument(format!(
                        "workflow component '{component}' {direction} port '{port}' declares \
                         required shape {:?} (rank {}), but the ONNX graph exposes rank {}",
                        contract.shape,
                        contract.rank(),
                        signature.rank()
                    )));
                }
                for (axis, (declared, actual)) in
                    contract.shape.iter().zip(&signature.shape).enumerate()
                {
                    if let (TensorDimension::Fixed(declared), PortDimension::Static(actual)) =
                        (declared, actual)
                        && usize::try_from(*declared).ok() != Some(*actual)
                    {
                        return Err(OrtError::InvalidArgument(format!(
                            "workflow component '{component}' {direction} port '{port}' required \
                             shape {:?}, but axis {axis} declares {declared} and the ONNX graph \
                             exposes {actual}",
                            contract.shape
                        )));
                    }
                }
            }
        }
    }
    validate_component_replacement_signatures(workflow, signatures)?;
    validate_workflow_invocations(&workflow.steps, workflow, signatures)?;
    Ok(())
}

fn validate_component_replacement_signatures(
    workflow: &onnx_genai_metadata::WorkflowSpec,
    signatures: &BTreeMap<String, ComponentSignature>,
) -> Result<()> {
    for (target_name, target) in &workflow.components {
        if !target.application_overridable {
            continue;
        }
        let target_contract = target.contract.as_ref().ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "overridable component '{target_name}' has no versioned contract"
            ))
        })?;
        let target_signature = signatures.get(target_name).ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "overridable component '{target_name}' has no inspected ONNX model"
            ))
        })?;
        for (replacement_name, replacement) in &workflow.components {
            if replacement_name == target_name {
                continue;
            }
            let Some(replacement_contract) = &replacement.contract else {
                continue;
            };
            if replacement_contract.id != target_contract.id
                || replacement_contract.version != target_contract.version
            {
                continue;
            }
            let Some(replacement_signature) = signatures.get(replacement_name) else {
                continue;
            };
            for (role, target_port) in &target_contract.bindings {
                let replacement_port = replacement_contract.bindings.get(role).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "replacement component '{replacement_name}' lacks semantic port '{role}' \
                         required by '{target_name}'"
                    ))
                })?;
                let (target_direction, target_port_signature) =
                    semantic_port_signature(target_name, role, target_port, target_signature)?;
                let (replacement_direction, replacement_port_signature) = semantic_port_signature(
                    replacement_name,
                    role,
                    replacement_port,
                    replacement_signature,
                )?;
                if target_direction != replacement_direction
                    || !port_signatures_compatible(
                        target_port_signature,
                        replacement_port_signature,
                    )
                {
                    return Err(OrtError::InvalidArgument(format!(
                        "replacement component '{replacement_name}' semantic port '{role}' is \
                         incompatible with '{target_name}'"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn port_signatures_compatible(left: &PortSignature, right: &PortSignature) -> bool {
    left.dtype == right.dtype
        && left.rank() == right.rank()
        && left
            .shape
            .iter()
            .zip(&right.shape)
            .all(|(left, right)| match (left, right) {
                (PortDimension::Static(left), PortDimension::Static(right)) => left == right,
                _ => true,
            })
}

fn semantic_port_signature<'a>(
    component: &str,
    role: &str,
    port: &str,
    signature: &'a ComponentSignature,
) -> Result<(&'static str, &'a PortSignature)> {
    match (signature.inputs.get(port), signature.outputs.get(port)) {
        (Some(signature), None) => Ok(("input", signature)),
        (None, Some(signature)) => Ok(("output", signature)),
        _ => Err(OrtError::InvalidArgument(format!(
            "component '{component}' semantic port '{role}' binds '{port}', which is not exactly \
             one ONNX input or output"
        ))),
    }
}

fn validate_workflow_invocations(
    steps: &[onnx_genai_metadata::WorkflowStep],
    workflow: &onnx_genai_metadata::WorkflowSpec,
    signatures: &BTreeMap<String, ComponentSignature>,
) -> Result<()> {
    for step in steps {
        match step {
            onnx_genai_metadata::WorkflowStep::Sequence { steps } => {
                validate_workflow_invocations(steps, workflow, signatures)?;
            }
            onnx_genai_metadata::WorkflowStep::Invoke {
                component,
                inputs,
                outputs,
            } => {
                let declaration = workflow.components.get(component).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "workflow invokes unknown component '{component}'"
                    ))
                })?;
                if declaration.application_overridable {
                    let contract = declaration.contract.as_ref().ok_or_else(|| {
                        OrtError::InvalidArgument(format!(
                            "overridable component '{component}' has no versioned contract"
                        ))
                    })?;
                    for port in inputs.keys().chain(outputs.keys()) {
                        if !contract.bindings.values().any(|binding| binding == port) {
                            return Err(OrtError::InvalidArgument(format!(
                                "overridable component '{component}' invocation port '{port}' is \
                                 not covered by its semantic contract ABI"
                            )));
                        }
                    }
                    for (replacement_name, replacement) in &workflow.components {
                        if replacement_name == component {
                            continue;
                        }
                        let Some(replacement_contract) = &replacement.contract else {
                            continue;
                        };
                        if replacement_contract.id != contract.id
                            || replacement_contract.version != contract.version
                        {
                            continue;
                        }
                        let Some(replacement_signature) = signatures.get(replacement_name) else {
                            continue;
                        };
                        let remapped_inputs = remap_invocation_ports(
                            component,
                            contract,
                            replacement_name,
                            replacement_contract,
                            inputs,
                        )?;
                        let remapped_outputs = remap_invocation_ports(
                            component,
                            contract,
                            replacement_name,
                            replacement_contract,
                            outputs,
                        )?;
                        validate_invocation_ports(
                            replacement_name,
                            "input",
                            &remapped_inputs,
                            &replacement_signature.inputs,
                            Some(&replacement_signature.defaulted_inputs),
                            true,
                        )?;
                        validate_invocation_ports(
                            replacement_name,
                            "output",
                            &remapped_outputs,
                            &replacement_signature.outputs,
                            None,
                            false,
                        )?;
                    }
                }
                if matches!(
                    declaration.implementation,
                    onnx_genai_metadata::ComponentImplementation::Onnx { .. }
                ) {
                    let signature = signatures.get(component).ok_or_else(|| {
                        OrtError::InvalidArgument(format!(
                            "workflow ONNX component '{component}' has no inspected model"
                        ))
                    })?;
                    validate_invocation_ports(
                        component,
                        "input",
                        inputs,
                        &signature.inputs,
                        Some(&signature.defaulted_inputs),
                        true,
                    )?;
                    validate_invocation_ports(
                        component,
                        "output",
                        outputs,
                        &signature.outputs,
                        None,
                        false,
                    )?;
                }
            }
            onnx_genai_metadata::WorkflowStep::Loop { setup, steps, .. } => {
                validate_workflow_invocations(setup, workflow, signatures)?;
                validate_workflow_invocations(steps, workflow, signatures)?;
            }
            onnx_genai_metadata::WorkflowStep::Branch { cases, default, .. } => {
                for case in cases.values() {
                    validate_workflow_invocations(
                        std::slice::from_ref(case),
                        workflow,
                        signatures,
                    )?;
                }
                if let Some(default) = default {
                    validate_workflow_invocations(
                        std::slice::from_ref(default.as_ref()),
                        workflow,
                        signatures,
                    )?;
                }
            }
            onnx_genai_metadata::WorkflowStep::Emit { .. } => {}
        }
    }
    Ok(())
}

fn remap_invocation_ports(
    target_name: &str,
    target_contract: &onnx_genai_metadata::ComponentContract,
    replacement_name: &str,
    replacement_contract: &onnx_genai_metadata::ComponentContract,
    ports: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    ports
        .iter()
        .map(|(port, value)| {
            let role = target_contract
                .bindings
                .iter()
                .find_map(|(role, bound)| (bound == port).then_some(role))
                .ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "overridable component '{target_name}' invocation port '{port}' is not \
                         covered by its semantic contract ABI"
                    ))
                })?;
            let replacement_port = replacement_contract.bindings.get(role).ok_or_else(|| {
                OrtError::InvalidArgument(format!(
                    "replacement component '{replacement_name}' lacks semantic port '{role}' \
                     required by '{target_name}'"
                ))
            })?;
            Ok((replacement_port.clone(), value.clone()))
        })
        .collect()
}

fn validate_invocation_ports(
    component: &str,
    direction: &str,
    bindings: &BTreeMap<String, String>,
    actual: &BTreeMap<String, PortSignature>,
    optional: Option<&BTreeSet<String>>,
    require_all: bool,
) -> Result<()> {
    for port in bindings.keys() {
        if !actual.contains_key(port) {
            return Err(OrtError::InvalidArgument(format!(
                "workflow invocation of '{component}' binds unknown {direction} port '{port}'"
            )));
        }
    }
    if require_all {
        for port in actual.keys() {
            if !bindings.contains_key(port) && !optional.is_some_and(|ports| ports.contains(port)) {
                return Err(OrtError::InvalidArgument(format!(
                    "workflow invocation of '{component}' is missing {direction} port '{port}'"
                )));
            }
        }
    }
    Ok(())
}

fn inspect_component_signatures(
    model_paths: &BTreeMap<String, PathBuf>,
) -> Result<BTreeMap<String, ComponentSignature>> {
    model_paths
        .iter()
        .map(|(component, path)| {
            inspect_component_signature(component, path)
                .map(|signature| (component.clone(), signature))
        })
        .collect()
}

fn inspect_component_signature(component: &str, path: &Path) -> Result<ComponentSignature> {
    let model = if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("textproto"))
    {
        let text = std::fs::read_to_string(path).map_err(|error| {
            component_inspection_error(
                component,
                path,
                format!("the ONNX textproto could not be read: {error}"),
            )
        })?;
        onnx_std::textproto::from_textproto(&text).map_err(|error| {
            component_inspection_error(
                component,
                path,
                format!("the ONNX textproto could not be parsed: {error}"),
            )
        })
    } else {
        onnx_std::load_model(path).map_err(|error| {
            component_inspection_error(
                component,
                path,
                format!("the ONNX model could not be loaded: {error}"),
            )
        })
    }?;
    // Admission must inspect the retained protobuf before scanning the execution
    // projection: graph_builder.rs:118-121 and 143-147 intentionally omit empty
    // GraphProto input/output names from the loaded IR.
    let source_proto = model.to_proto().map_err(|error| {
        component_inspection_error(
            component,
            path,
            format!("the retained ONNX protobuf could not be inspected: {error}"),
        )
    })?;
    let source_graph = source_proto.graph.as_ref().ok_or_else(|| {
        component_inspection_error(
            component,
            path,
            "the retained ONNX protobuf has no graph".to_string(),
        )
    })?;
    if source_graph.input.iter().any(|input| input.name.is_empty()) {
        return Err(OrtError::InvalidArgument(format!(
            "package admission rejected component '{component}': an ONNX graph input is \
             unnamed at model path '{}', so the pipeline cannot bind it. How to fix: \
             regenerate the graph with explicit input names and a matching native sidecar",
            path.display()
        )));
    }
    if source_graph
        .output
        .iter()
        .any(|output| output.name.is_empty())
    {
        return Err(OrtError::InvalidArgument(format!(
            "package admission rejected component '{component}': an ONNX graph output is \
             unnamed at model path '{}', so dataflow cannot reference it. How to fix: \
             regenerate the graph with explicit output names and a matching native sidecar",
            path.display()
        )));
    }

    let mut signature = ComponentSignature::default();
    let initializer_names = source_graph
        .initializer
        .iter()
        .map(|initializer| initializer.name.as_str())
        .collect::<BTreeSet<_>>();
    // Retain the target's initializer inventory so a folded-carry
    // `token_embedding.table` can be resolved against the exact set of weights
    // the model owns, rather than trusting the declared string.
    signature.initializers = initializer_names
        .iter()
        .map(|name| name.to_string())
        .collect();

    for input in &source_graph.input {
        let name = input.name.clone();
        if initializer_names.contains(name.as_str()) {
            signature.defaulted_inputs.insert(name.clone());
        }
        signature
            .inputs
            .insert(name, raw_input_signature(component, path, input)?);
    }

    for output in &model.graph.outputs {
        let value = model.graph.value(*output);
        let name = value
            .name
            .clone()
            .expect("validated GraphProto output names survive loader projection");
        signature.outputs.insert(
            name,
            PortSignature {
                dtype: value.dtype,
                shape: value
                    .shape
                    .iter()
                    .map(|dimension| match dimension {
                        Dim::Static(value) => PortDimension::Static(*value),
                        Dim::Symbolic(_) => PortDimension::Dynamic,
                    })
                    .collect(),
            },
        );
    }

    Ok(signature)
}

fn raw_input_signature(
    component: &str,
    path: &Path,
    input: &ValueInfoProto,
) -> Result<PortSignature> {
    let tensor = input
        .r#type
        .as_ref()
        .and_then(|input_type| input_type.value.as_ref())
        .and_then(|input_type| match input_type {
            type_proto::Value::TensorType(tensor) => Some(tensor),
            _ => None,
        })
        .ok_or_else(|| {
            component_inspection_error(
                component,
                path,
                format!(
                    "ONNX graph input '{}' does not declare a tensor type",
                    input.name
                ),
            )
        })?;
    let dtype = DataType::from_onnx(tensor.elem_type).ok_or_else(|| {
        component_inspection_error(
            component,
            path,
            format!(
                "ONNX graph input '{}' declares unsupported tensor dtype {}",
                input.name, tensor.elem_type
            ),
        )
    })?;
    let shape = tensor
        .shape
        .as_ref()
        .map(|shape| {
            shape
                .dim
                .iter()
                .map(|dimension| match dimension.value.as_ref() {
                    Some(tensor_shape_proto::dimension::Value::DimValue(value)) if *value >= 0 => {
                        PortDimension::Static(*value as usize)
                    }
                    _ => PortDimension::Dynamic,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(PortSignature { dtype, shape })
}

fn component_inspection_error(component: &str, path: &Path, cause: String) -> OrtError {
    OrtError::InvalidArgument(format!(
        "package admission rejected component '{component}': {cause} at model path '{}'. \
         How to fix: regenerate the package with a valid ONNX graph and native sidecar for \
         component '{component}'",
        path.display()
    ))
}

fn parse_dtype(value: &str) -> Option<DataType> {
    Some(match value.trim().to_ascii_lowercase().as_str() {
        "float" | "float32" | "fp32" | "f32" => DataType::Float32,
        "float16" | "fp16" | "f16" => DataType::Float16,
        "bfloat16" | "bf16" => DataType::BFloat16,
        "float64" | "fp64" | "f64" | "double" => DataType::Float64,
        "int64" | "i64" => DataType::Int64,
        "int32" | "i32" => DataType::Int32,
        "int16" | "i16" => DataType::Int16,
        "int8" | "i8" => DataType::Int8,
        "uint64" | "u64" => DataType::Uint64,
        "uint32" | "u32" => DataType::Uint32,
        "uint16" | "u16" => DataType::Uint16,
        "uint8" | "u8" => DataType::Uint8,
        "bool" | "boolean" => DataType::Bool,
        "string" => DataType::String,
        "float8_e4m3fn" | "fp8_e4m3fn" => DataType::Float8E4M3FN,
        "float8_e4m3fnuz" | "fp8_e4m3fnuz" => DataType::Float8E4M3FNUZ,
        "float8_e5m2" | "fp8_e5m2" => DataType::Float8E5M2,
        "float8_e5m2fnuz" | "fp8_e5m2fnuz" => DataType::Float8E5M2FNUZ,
        "float8_e8m0" | "fp8_e8m0" => DataType::Float8E8M0,
        "float4_e2m1" | "fp4_e2m1" => DataType::Float4E2M1,
        "int4" | "i4" => DataType::Int4,
        "uint4" | "u4" => DataType::Uint4,
        "int2" | "i2" => DataType::Int2,
        "uint2" | "u2" => DataType::Uint2,
        "complex64" => DataType::Complex64,
        "complex128" => DataType::Complex128,
        _ => return None,
    })
}

fn dtype_name(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Undefined => "undefined",
        DataType::Float32 => "float32",
        DataType::Uint8 => "uint8",
        DataType::Int8 => "int8",
        DataType::Uint16 => "uint16",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::String => "string",
        DataType::Bool => "bool",
        DataType::Float16 => "float16",
        DataType::Float64 => "float64",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Complex64 => "complex64",
        DataType::Complex128 => "complex128",
        DataType::BFloat16 => "bfloat16",
        DataType::Float8E4M3FN => "float8_e4m3fn",
        DataType::Float8E4M3FNUZ => "float8_e4m3fnuz",
        DataType::Float8E5M2 => "float8_e5m2",
        DataType::Float8E5M2FNUZ => "float8_e5m2fnuz",
        DataType::Uint4 => "uint4",
        DataType::Int4 => "int4",
        DataType::Float4E2M1 => "float4_e2m1",
        DataType::Float8E8M0 => "float8_e8m0",
        DataType::Uint2 => "uint2",
        DataType::Int2 => "int2",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the real speculative verifier ONNX fixture, whose graph owns the
    /// inline initializers `const_1d_0` and `const_1d_1`.
    fn verifier_model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../tests/fixtures/onnx_genai_workflows/speculative/verifier/model.onnx.textproto",
        )
    }

    /// Inspect the real verifier fixture into a signature map keyed `verifier`,
    /// exactly as pipeline admission does at load time.
    fn verifier_signatures() -> BTreeMap<String, ComponentSignature> {
        let mut model_paths = BTreeMap::new();
        model_paths.insert("verifier".to_string(), verifier_model_path());
        inspect_component_signatures(&model_paths).expect("verifier fixture should inspect")
    }

    /// Build a chained-proposer speculative contract whose folded-carry
    /// `token_embedding` names the verifier component and the given table.
    fn chained_speculative_with_table(table: &str) -> onnx_genai_metadata::SpeculativeContract {
        let yaml = [
            "identity: onnx-genai.speculative".to_string(),
            "version: '1'".to_string(),
            "proposer: proposer".to_string(),
            "target: verifier".to_string(),
            "proposal_execution:".to_string(),
            "  kind: chained".to_string(),
            "  token_embedding_input: inputs_embeds".to_string(),
            "  logits_output: draft_logits".to_string(),
            "  token_embedding:".to_string(),
            "    component: verifier".to_string(),
            format!("    table: {table}"),
            "vocabulary:".to_string(),
            "  kind: identical".to_string(),
            "max_proposal_width: 4".to_string(),
            "verification:".to_string(),
            "  target_output: {component: verifier, output: logits}".to_string(),
            "  accepted_path: {kind: runtime, binding: accepted_prefix}".to_string(),
        ]
        .join("\n");
        serde_yaml::from_str(&yaml).expect("speculative contract YAML should parse")
    }

    #[test]
    fn token_embedding_table_resolving_to_a_real_initializer_is_admitted() {
        let signatures = verifier_signatures();
        // The inventory was really read from the ONNX artifact, not assumed.
        let inventory = &signatures
            .get("verifier")
            .expect("verifier signature")
            .initializers;
        assert!(inventory.contains("const_1d_0"));
        assert!(inventory.contains("const_1d_1"));

        let speculative = chained_speculative_with_table("const_1d_0");
        validate_speculative_shared_initializers(&speculative, &signatures)
            .expect("a table matching a real initializer must be admitted");
    }

    #[test]
    fn a_nonexistent_token_embedding_table_is_rejected_at_admission() {
        let signatures = verifier_signatures();
        // Non-empty and structurally valid, but absent from the target model.
        let speculative = chained_speculative_with_table("model.embed_tokens.weight");
        let err = validate_speculative_shared_initializers(&speculative, &signatures)
            .expect_err("a dangling embedding table must be rejected fail-closed");
        let message = err.to_string();
        assert!(
            message.contains("model.embed_tokens.weight"),
            "error must name the offending table: {message}"
        );
        assert!(
            message.contains("verifier"),
            "error must name the target component: {message}"
        );
    }

    #[test]
    fn a_token_embedding_component_without_an_inspected_model_is_rejected() {
        let signatures = verifier_signatures();
        let mut speculative = chained_speculative_with_table("const_1d_0");
        // Point the embedding at a component that was never inspected/loaded.
        if let onnx_genai_metadata::SpeculativeProposalExecution::Chained {
            token_embedding: Some(embedding),
            ..
        } = &mut speculative.proposal_execution
        {
            embedding.component = "ghost".to_string();
        }
        let err = validate_speculative_shared_initializers(&speculative, &signatures)
            .expect_err("an embedding component with no inspected model must be rejected");
        assert!(err.to_string().contains("ghost"));
    }

    #[test]
    fn a_block_proposer_without_a_folded_carry_is_ignored() {
        let signatures = verifier_signatures();
        let yaml = [
            "identity: onnx-genai.speculative",
            "version: '1'",
            "proposer: proposer",
            "target: verifier",
            "proposal_execution:",
            "  kind: block",
            "vocabulary:",
            "  kind: identical",
            "max_proposal_width: 4",
            "verification:",
            "  target_output: {component: verifier, output: logits}",
            "  accepted_path: {kind: runtime, binding: accepted_prefix}",
        ]
        .join("\n");
        let speculative: onnx_genai_metadata::SpeculativeContract =
            serde_yaml::from_str(&yaml).expect("block speculative contract should parse");
        validate_speculative_shared_initializers(&speculative, &signatures)
            .expect("a proposer without a folded-carry token_embedding needs no inventory check");
    }
}
