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
}

pub(crate) fn validate_pipeline_admission(
    spec: &PipelineSpec,
    model_paths: &BTreeMap<String, PathBuf>,
) -> Result<()> {
    let signatures = inspect_component_signatures(model_paths)?;
    validate_workflow_signatures(&spec.workflow, &signatures)
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
                if contract.rank != signature.rank() {
                    return Err(OrtError::InvalidArgument(format!(
                        "workflow component '{component}' {direction} '{port}' declares rank {}, \
                         but the ONNX graph exposes rank {}",
                        contract.rank,
                        signature.rank()
                    )));
                }
                if let Some(shape) = &contract.shape {
                    for (axis, (declared, actual)) in shape.iter().zip(&signature.shape).enumerate()
                    {
                        if let (TensorDimension::Fixed(declared), PortDimension::Static(actual)) =
                            (declared, actual)
                            && usize::try_from(*declared).ok() != Some(*actual)
                        {
                            return Err(OrtError::InvalidArgument(format!(
                                "workflow component '{component}' {direction} '{port}' axis \
                                 {axis} declares {declared}, but the ONNX graph exposes {actual}"
                            )));
                        }
                    }
                }
            }
        }
    }
    validate_workflow_invocations(&workflow.steps, workflow, signatures)?;
    Ok(())
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
