use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use onnx_genai_metadata::{
    PipelineSpec, PipelineStrategy, PipelineStrategyKind, PreprocessingSpec,
};
use onnx_std::ir::{DataType, Dim, SymbolId};

use crate::{OrtError, Result};

#[derive(Debug, Clone)]
struct PortSignature {
    dtype: DataType,
    shape: Vec<Dim>,
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
    preprocessing: Option<&PreprocessingSpec>,
    model_paths: &BTreeMap<String, PathBuf>,
) -> Result<()> {
    let signatures = inspect_component_signatures(model_paths)?;
    validate_edges(spec, &signatures)?;

    let preprocessed_inputs = validate_image_program(spec, preprocessing, &signatures)?;
    validate_input_closure(spec, &signatures, &preprocessed_inputs)
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

    let graph = &model.graph;
    let mut signature = ComponentSignature::default();

    for input in &graph.inputs {
        let value = graph.value(*input);
        let name = value
            .name
            .clone()
            .expect("validated GraphProto input names survive loader projection");
        if graph.initializers.contains_key(input) {
            signature.defaulted_inputs.insert(name.clone());
        }
        signature.inputs.insert(
            name,
            PortSignature {
                dtype: value.dtype,
                shape: value.shape.clone(),
            },
        );
    }

    for output in &graph.outputs {
        let value = graph.value(*output);
        let name = value
            .name
            .clone()
            .expect("validated GraphProto output names survive loader projection");
        signature.outputs.insert(
            name,
            PortSignature {
                dtype: value.dtype,
                shape: value.shape.clone(),
            },
        );
    }

    Ok(signature)
}

fn validate_edges(
    spec: &PipelineSpec,
    signatures: &BTreeMap<String, ComponentSignature>,
) -> Result<()> {
    let synthetic_outputs = synthetic_outputs(&spec.strategy);
    let transformed_loop_components = transformed_loop_components(&spec.strategy);
    for edge in &spec.dataflow {
        let (source_component, source_port) = parse_endpoint(&edge.from)?;
        let (destination_component, destination_port) = parse_endpoint(&edge.to)?;
        let source = signatures
            .get(source_component)
            .and_then(|signature| signature.outputs.get(source_port))
            .or_else(|| synthetic_outputs.get(&(source_component, source_port)))
            .ok_or_else(|| {
                admission_error(
                    &edge.from,
                    format!(
                        "dataflow source '{}' is not an ONNX graph output",
                        edge.from
                    ),
                    format!(
                        "regenerate the native sidecar so {} names a real producer output",
                        edge.from
                    ),
                )
            })?;
        let destination = signatures
            .get(destination_component)
            .and_then(|signature| signature.inputs.get(destination_port))
            .ok_or_else(|| {
                admission_error(
                    &edge.to,
                    format!(
                        "dataflow destination '{}' is not an ONNX graph input",
                        edge.to
                    ),
                    format!(
                        "regenerate the native sidecar so {} names a real consumer input",
                        edge.to
                    ),
                )
            })?;

        if source_component == destination_component
            && transformed_loop_components.contains(source_component)
        {
            continue;
        }

        if source.dtype != destination.dtype {
            return Err(admission_error(
                &edge.to,
                format!(
                    "dataflow edge '{} -> {}' has incompatible dtypes: producer is {}, consumer is {}{}",
                    edge.from,
                    edge.to,
                    dtype_name(source.dtype),
                    dtype_name(destination.dtype),
                    edge.dtype
                        .as_deref()
                        .map(|dtype| format!(", metadata declares '{dtype}'"))
                        .unwrap_or_default()
                ),
                format!(
                    "regenerate the native sidecar or graphs so {} and {} use the same dtype",
                    edge.from, edge.to
                ),
            ));
        }

        if source.rank() != destination.rank() {
            return Err(admission_error(
                &edge.to,
                format!(
                    "dataflow edge '{} -> {}' has incompatible ranks: producer rank {}, consumer rank {}",
                    edge.from,
                    edge.to,
                    source.rank(),
                    destination.rank()
                ),
                format!(
                    "regenerate the native sidecar or add an explicit transform so {} matches {}",
                    edge.from, edge.to
                ),
            ));
        }
        if let Some(declared) = edge.dtype.as_deref() {
            let declared_dtype = parse_dtype(declared).ok_or_else(|| {
                admission_error(
                    &edge.to,
                    format!(
                        "dataflow edge '{} -> {}' declares unsupported dtype '{declared}'",
                        edge.from, edge.to
                    ),
                    "regenerate the native sidecar with the producer's canonical tensor dtype"
                        .to_string(),
                )
            })?;
            if declared_dtype != source.dtype {
                return Err(admission_error(
                    &edge.to,
                    format!(
                        "dataflow edge '{} -> {}' declares dtype '{declared}', but both graphs use {}",
                        edge.from,
                        edge.to,
                        dtype_name(source.dtype)
                    ),
                    "regenerate the native sidecar so the edge dtype matches both graph ports"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn transformed_loop_components(strategy: &PipelineStrategy) -> BTreeSet<&str> {
    let mut components = BTreeSet::new();
    collect_transformed_loop_components(strategy, &mut components);
    components
}

fn collect_transformed_loop_components<'a>(
    strategy: &'a PipelineStrategy,
    components: &mut BTreeSet<&'a str>,
) {
    match strategy.kind {
        PipelineStrategyKind::Iterative => {
            if strategy.scheduler_config.is_some()
                && let Some(denoiser) = strategy.denoiser.as_deref()
            {
                components.insert(denoiser);
            }
        }
        PipelineStrategyKind::Composite => {
            for stage in &strategy.stages {
                collect_transformed_loop_components(&stage.strategy, components);
            }
        }
        _ => {}
    }
}

fn validate_image_program(
    spec: &PipelineSpec,
    preprocessing: Option<&PreprocessingSpec>,
    signatures: &BTreeMap<String, ComponentSignature>,
) -> Result<BTreeSet<String>> {
    let image_program = preprocessing.and_then(|preprocessing| preprocessing.image.as_ref());
    let mut bound = BTreeSet::new();

    if let Some(program) = image_program {
        for output in &program.outputs {
            let endpoint = resolve_input_endpoint(&output.name, signatures)?;
            let Some((endpoint, input)) = endpoint else {
                if output.optional.unwrap_or(false) {
                    continue;
                }
                return Err(admission_error(
                    &output.name,
                    format!(
                        "required preprocessing.image output '{}' does not resolve to an ONNX component input",
                        output.name
                    ),
                    format!(
                        "regenerate the native sidecar so preprocessing output '{}' names an exact component.port",
                        output.name
                    ),
                ));
            };
            let declared_dtype = parse_dtype(&output.dtype).ok_or_else(|| {
                admission_error(
                    &endpoint,
                    format!(
                        "preprocessing.image output '{}' declares unsupported dtype '{}'",
                        output.name, output.dtype
                    ),
                    "regenerate the native sidecar with a supported tensor dtype".to_string(),
                )
            })?;
            if declared_dtype != input.dtype {
                return Err(admission_error(
                    &endpoint,
                    format!(
                        "preprocessing.image output '{}' declares {}, but {} expects {}",
                        output.name,
                        dtype_name(declared_dtype),
                        endpoint,
                        dtype_name(input.dtype)
                    ),
                    format!(
                        "regenerate the native sidecar so preprocessing output '{}' matches {}",
                        output.name, endpoint
                    ),
                ));
            }
            if !bound.insert(endpoint.clone()) {
                return Err(admission_error(
                    &endpoint,
                    "multiple preprocessing.image outputs bind the same ONNX input".to_string(),
                    "regenerate the native sidecar so every preprocessing output binds a unique component.port"
                        .to_string(),
                ));
            }
        }
    }

    if image_program.is_some()
        && !decoder_components(&spec.strategy).is_empty()
        && spec.vision.is_none()
        && let Some(endpoint) = bound.first()
    {
        return Err(admission_error(
            endpoint,
            "the image preprocessing endpoint is incomplete: preprocessing.image is declared for an autoregressive pipeline, but pipeline.vision has no prompt expansion program"
                .to_string(),
            "regenerate the native sidecar with a pipeline.vision expansion contract that matches the preprocessing program"
                .to_string(),
        ));
    }

    if spec.vision.is_some() {
        for (component, model) in &spec.models {
            if model.role != "vision_encoder" {
                continue;
            }
            let signature = signatures
                .get(component)
                .expect("inspected declared component");
            for port in signature.inputs.keys() {
                let endpoint = format!("{component}.{port}");
                let has_edge = spec.dataflow.iter().any(|edge| edge.to == endpoint);
                if !has_edge && !bound.contains(&endpoint) {
                    return Err(admission_error(
                        &endpoint,
                        "the declared image modality endpoint cannot be constructed: pipeline.vision is present, but preprocessing.image does not bind this required ONNX input"
                            .to_string(),
                        format!(
                            "regenerate the native sidecar with preprocessing.image output '{}' and a matching vision expansion program",
                            endpoint
                        ),
                    ));
                }
            }
        }
    }

    Ok(bound)
}

fn validate_input_closure(
    spec: &PipelineSpec,
    signatures: &BTreeMap<String, ComponentSignature>,
    preprocessed_inputs: &BTreeSet<String>,
) -> Result<()> {
    let decoders = decoder_components(&spec.strategy);
    let generated = generated_inputs(spec, &decoders);

    for (component, signature) in signatures {
        let explicit_decoder_contract = decoders.contains(component)
            && spec
                .models
                .get(component)
                .is_some_and(|model| model.io.is_some());
        for port in signature.inputs.keys() {
            let endpoint = format!("{component}.{port}");
            let incoming_edges = spec
                .dataflow
                .iter()
                .filter(|edge| edge.to == endpoint)
                .count();
            let defaulted = signature.defaulted_inputs.contains(port);
            let generated_or_stateful = generated.contains(&endpoint);
            let preprocessed = preprocessed_inputs.contains(&endpoint);
            // Requests may bind any component.port. Without an explicit decoder I/O contract,
            // absence of another source is not proof that the port is unbound, so fail open.
            let external = !explicit_decoder_contract
                && incoming_edges == 0
                && !defaulted
                && !generated_or_stateful
                && !preprocessed;
            let binding_count = incoming_edges
                + usize::from(defaulted)
                + usize::from(generated_or_stateful)
                + usize::from(preprocessed)
                + usize::from(external);

            if binding_count == 0 {
                return Err(admission_error(
                    &endpoint,
                    "required ONNX graph input is unbound: no external, generated, stateful, default, preprocessing, or dataflow source is declared for this port"
                        .to_string(),
                    format!(
                        "regenerate the native sidecar so {endpoint} is fed by exactly one declared source"
                    ),
                ));
            }
            if binding_count > 1 {
                return Err(admission_error(
                    &endpoint,
                    "required ONNX graph input has multiple binding sources, so execution would be ambiguous"
                        .to_string(),
                    format!(
                        "regenerate the native sidecar so {endpoint} is fed by exactly one external, generated, stateful, default, or dataflow source"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn generated_inputs(spec: &PipelineSpec, decoders: &BTreeSet<String>) -> BTreeSet<String> {
    let mut generated = BTreeSet::new();

    for (component, model) in &spec.models {
        if let Some(io) = model.io.as_ref() {
            for port in [
                io.token_input.as_deref(),
                decoders
                    .contains(component)
                    .then_some(io.attention_mask_input.as_deref())
                    .flatten(),
                decoders
                    .contains(component)
                    .then_some(io.position_ids_input.as_deref())
                    .flatten(),
            ]
            .into_iter()
            .flatten()
            {
                generated.insert(format!("{component}.{port}"));
            }
            for port in io.kv_inputs.iter().flatten() {
                generated.insert(format!("{component}.{port}"));
            }
            for port in io.cross_kv_inputs.iter().flatten() {
                generated.insert(format!("{component}.{port}"));
            }
            for pair in io.state_pairs.iter().flatten() {
                generated.insert(format!("{component}.{}", pair.input));
            }
        }
    }

    if let Some(position) = spec.positions.as_ref() {
        for decoder in decoders {
            generated.insert(format!("{decoder}.{}", position.input));
        }
    }
    collect_strategy_generated_inputs(&spec.strategy, &mut generated);
    generated
}

fn collect_strategy_generated_inputs(
    strategy: &PipelineStrategy,
    generated: &mut BTreeSet<String>,
) {
    match strategy.kind {
        PipelineStrategyKind::Iterative => {
            if let (Some(denoiser), Some(timestep)) = (
                strategy.denoiser.as_deref(),
                strategy.timestep_input.as_deref(),
            ) {
                generated.insert(format!("{denoiser}.{timestep}"));
            }
        }
        PipelineStrategyKind::NestedAutoregressive => {
            if let Some(pre_embedder) = strategy.pre_embedder.as_ref() {
                generated.insert(format!(
                    "{}.{}",
                    pre_embedder.component, pre_embedder.frame_codes_input
                ));
                if let Some(text_embed) = pre_embedder.text_embed_input.as_deref() {
                    generated.insert(format!("{}.{}", pre_embedder.component, text_embed));
                }
            }
            if let Some(prefill) = strategy.prefill_embedder.as_ref() {
                generated.insert(format!("{}.{}", prefill.component, prefill.prompt_input));
            }
        }
        PipelineStrategyKind::Composite => {
            for stage in &strategy.stages {
                collect_strategy_generated_inputs(&stage.strategy, generated);
            }
        }
        _ => {}
    }
}

fn synthetic_outputs(strategy: &PipelineStrategy) -> BTreeMap<(&str, &str), PortSignature> {
    let mut outputs = BTreeMap::new();
    collect_synthetic_outputs(strategy, &mut outputs);
    outputs
}

fn collect_synthetic_outputs<'a>(
    strategy: &'a PipelineStrategy,
    outputs: &mut BTreeMap<(&'a str, &'a str), PortSignature>,
) {
    match strategy.kind {
        PipelineStrategyKind::Autoregressive => {
            if let Some(decoder) = strategy.decoder.as_deref() {
                outputs.insert(
                    (decoder, "output_ids"),
                    PortSignature {
                        dtype: DataType::Int64,
                        shape: vec![Dim::Static(1), Dim::Symbolic(SymbolId(u32::MAX))],
                    },
                );
            }
        }
        PipelineStrategyKind::NestedAutoregressive => {
            if let Some(outer) = strategy.outer.as_deref() {
                outputs.insert(
                    (outer, "output_codes"),
                    PortSignature {
                        dtype: DataType::Int64,
                        shape: vec![
                            Dim::Static(1),
                            Dim::Symbolic(SymbolId(u32::MAX)),
                            Dim::Symbolic(SymbolId(u32::MAX - 1)),
                        ],
                    },
                );
            }
        }
        PipelineStrategyKind::Composite => {
            for stage in &strategy.stages {
                collect_synthetic_outputs(&stage.strategy, outputs);
            }
        }
        _ => {}
    }
}

fn decoder_components(strategy: &PipelineStrategy) -> BTreeSet<String> {
    let mut decoders = BTreeSet::new();
    collect_decoder_components(strategy, &mut decoders);
    decoders
}

fn collect_decoder_components(strategy: &PipelineStrategy, decoders: &mut BTreeSet<String>) {
    match strategy.kind {
        PipelineStrategyKind::Autoregressive => {
            if let Some(decoder) = strategy.decoder.as_ref() {
                decoders.insert(decoder.clone());
            }
        }
        PipelineStrategyKind::NestedAutoregressive => {
            if let Some(outer) = strategy.outer.as_ref() {
                decoders.insert(outer.clone());
            }
            if let Some(inner) = strategy.inner.as_ref() {
                decoders.insert(inner.clone());
            }
        }
        PipelineStrategyKind::Composite => {
            for stage in &strategy.stages {
                collect_decoder_components(&stage.strategy, decoders);
            }
        }
        _ => {}
    }
}

fn resolve_input_endpoint<'a>(
    endpoint: &str,
    signatures: &'a BTreeMap<String, ComponentSignature>,
) -> Result<Option<(String, &'a PortSignature)>> {
    if let Some((component, port)) = parse_endpoint_unchecked(endpoint) {
        return Ok(signatures
            .get(component)
            .and_then(|signature| signature.inputs.get(port))
            .map(|input| (endpoint.to_string(), input)));
    }

    let matches = signatures
        .iter()
        .filter_map(|(component, signature)| {
            signature
                .inputs
                .get(endpoint)
                .map(|input| (format!("{component}.{endpoint}"), input))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [(endpoint, input)] => Ok(Some((endpoint.clone(), *input))),
        _ => Err(admission_error(
            endpoint,
            format!(
                "preprocessing endpoint '{endpoint}' is ambiguous across {} ONNX inputs",
                matches.len()
            ),
            "regenerate the native sidecar with an exact component.port preprocessing output"
                .to_string(),
        )),
    }
}

fn parse_endpoint(endpoint: &str) -> Result<(&str, &str)> {
    parse_endpoint_unchecked(endpoint).ok_or_else(|| {
        OrtError::InvalidArgument(format!(
            "package admission rejected endpoint '{endpoint}': expected component.port. \
             Regenerate the native sidecar with exact component.port dataflow endpoints"
        ))
    })
}

fn parse_endpoint_unchecked(endpoint: &str) -> Option<(&str, &str)> {
    let (component, port) = endpoint.split_once('.')?;
    (!component.is_empty() && !port.is_empty()).then_some((component, port))
}

fn admission_error(endpoint: &str, why: String, fix: String) -> OrtError {
    OrtError::InvalidArgument(format!(
        "package admission rejected {endpoint}: {why}. How to fix: {fix}"
    ))
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
