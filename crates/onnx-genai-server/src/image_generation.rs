use onnx_genai::GenerateRequest;
use onnx_genai_engine::PipelineGenerateRequest;
use onnx_genai_metadata::{
    LiteralValue, PipelineSpec, RuntimeInputRole, ScalarValue, SemanticInputRole, TensorContract,
    WorkflowOutputRole,
};
use onnx_genai_ort::{DataType, Value};

#[derive(Debug, Clone)]
pub(crate) struct ImageInputBinding {
    pub(crate) name: String,
    pub(crate) contract: TensorContract,
    pub(crate) default: Option<LiteralValue>,
}

#[derive(Debug, Clone)]
pub(crate) struct ImagePipelineSpec {
    pub(crate) prompt_tokens: Option<ImageInputBinding>,
    pub(crate) negative_prompt_tokens: Option<ImageInputBinding>,
    pub(crate) seed: Option<ImageInputBinding>,
    pub(crate) steps: Option<ImageInputBinding>,
    pub(crate) guidance_scale: Option<ImageInputBinding>,
    pub(crate) width: Option<ImageInputBinding>,
    pub(crate) height: Option<ImageInputBinding>,
    pub(crate) media: Option<ImageInputBinding>,
    pub(crate) denoising_strength: Option<ImageInputBinding>,
    pub(crate) samplers: Vec<String>,
}

impl ImagePipelineSpec {
    pub(crate) fn from_pipeline(spec: &PipelineSpec) -> Option<Self> {
        spec.workflow
            .outputs
            .values()
            .any(|output| output.role == WorkflowOutputRole::Image)
            .then(|| Self {
                prompt_tokens: binding(spec, RuntimeInputRole::PromptTokens),
                negative_prompt_tokens: binding(spec, RuntimeInputRole::NegativePromptTokens),
                seed: binding(spec, RuntimeInputRole::Seed),
                steps: binding(spec, RuntimeInputRole::MaxIterations),
                guidance_scale: binding(spec, RuntimeInputRole::GuidanceScale),
                width: binding(spec, RuntimeInputRole::Width),
                height: binding(spec, RuntimeInputRole::Height),
                media: binding(spec, RuntimeInputRole::Media).filter(|binding| {
                    binding.contract.dtype == "uint8" && binding.contract.rank == 1
                }),
                denoising_strength: binding(spec, RuntimeInputRole::DenoisingStrength),
                samplers: declared_samplers(spec),
            })
    }

    pub(crate) fn default_i64(binding: Option<&ImageInputBinding>) -> Option<i64> {
        match binding?.default.as_ref()? {
            LiteralValue::Scalar(ScalarValue::Integer(value)) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn default_f32(binding: Option<&ImageInputBinding>) -> Option<f32> {
        match binding?.default.as_ref()? {
            LiteralValue::Scalar(ScalarValue::Float(value)) => Some(*value as f32),
            LiteralValue::Scalar(ScalarValue::Integer(value)) => Some(*value as f32),
            _ => None,
        }
    }
}

fn binding(spec: &PipelineSpec, wanted: RuntimeInputRole) -> Option<ImageInputBinding> {
    spec.workflow.inputs.iter().find_map(|(name, input)| {
        matches!(
            &input.role,
            SemanticInputRole::Runtime { role, .. } if *role == wanted
        )
        .then(|| ImageInputBinding {
            name: name.clone(),
            contract: input.contract.clone(),
            default: input.default.clone(),
        })
    })
}

fn declared_samplers(spec: &PipelineSpec) -> Vec<String> {
    let mut samplers = spec
        .workflow
        .components
        .values()
        .filter_map(|component| component.contract.as_ref())
        .filter(|contract| contract.id == "onnx-genai.diffusion-schedule")
        .filter_map(|contract| match contract.parameters.get("solver") {
            Some(ScalarValue::String(value)) => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    samplers.sort();
    samplers.dedup();
    samplers
}

#[derive(Debug)]
pub(crate) struct ProducedImage {
    pub(crate) values: Vec<f32>,
    pub(crate) shape: Vec<i64>,
}

#[derive(Debug)]
pub(crate) struct ImageExecutionRequest {
    pub(crate) request: GenerateRequest,
    pub(crate) inputs: Vec<(String, ImageInputValue)>,
}

impl ImageExecutionRequest {
    pub(crate) fn into_pipeline(self) -> anyhow::Result<PipelineGenerateRequest> {
        let mut request = PipelineGenerateRequest::new(self.request);
        for (name, input) in self.inputs {
            request = request.with_input(name, input.into_value()?);
        }
        Ok(request)
    }
}

#[derive(Debug)]
pub(crate) enum ImageInputValue {
    I64 { values: Vec<i64>, shape: Vec<i64> },
    F32 { values: Vec<f32>, shape: Vec<i64> },
    Bytes(Vec<u8>),
}

impl ImageInputValue {
    fn into_value(self) -> anyhow::Result<Value> {
        match self {
            Self::I64 { values, shape } => {
                Value::from_slice_i64(&values, &shape).map_err(Into::into)
            }
            Self::F32 { values, shape } => {
                Value::from_slice_f32(&values, &shape).map_err(Into::into)
            }
            Self::Bytes(bytes) => {
                let length = i64::try_from(bytes.len())?;
                Value::from_raw_bytes(bytes, &[length], DataType::Uint8).map_err(Into::into)
            }
        }
    }
}

pub(crate) fn scalar_i64(
    binding: &ImageInputBinding,
    value: i64,
) -> anyhow::Result<ImageInputValue> {
    let shape = scalar_shape(&binding.contract)?;
    Ok(ImageInputValue::I64 {
        values: vec![value; numel(&shape)],
        shape,
    })
}

pub(crate) fn scalar_f32(
    binding: &ImageInputBinding,
    value: f32,
) -> anyhow::Result<ImageInputValue> {
    let shape = scalar_shape(&binding.contract)?;
    Ok(ImageInputValue::F32 {
        values: vec![value; numel(&shape)],
        shape,
    })
}

fn scalar_shape(contract: &TensorContract) -> anyhow::Result<Vec<i64>> {
    match contract.rank {
        0 => Ok(Vec::new()),
        1 => Ok(vec![1]),
        rank => anyhow::bail!("image API scalar binding requires rank 0 or 1, got rank {rank}"),
    }
}

fn numel(shape: &[i64]) -> usize {
    shape.iter().product::<i64>().max(1) as usize
}
