use anyhow::Context;
use onnx_genai::{GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_engine::PipelineGenerateRequest;
use onnx_genai_metadata::{
    LiteralValue, PipelineSpec, PixelValueRange, RuntimeInputRole, ScalarValue, SemanticInputRole,
    TensorContract, WorkflowInputSource, WorkflowOutputRole,
};
use onnx_genai_ort::{DataType, Tokenizer, Value};

#[derive(Debug, Clone)]
pub(crate) struct ImageInputBinding {
    pub(crate) name: String,
    pub(crate) contract: TensorContract,
    pub(crate) default: Option<LiteralValue>,
    pub(crate) required: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ImagePipelineSpec {
    pub(crate) output_value_range: PixelValueRange,
    pub(crate) prompt_tokens: Option<ImageInputBinding>,
    pub(crate) negative_prompt_tokens: Option<ImageInputBinding>,
    pub(crate) seed: Option<ImageInputBinding>,
    pub(crate) steps: Option<ImageInputBinding>,
    pub(crate) guidance_scale: Option<ImageInputBinding>,
    pub(crate) width: Option<ImageInputBinding>,
    pub(crate) height: Option<ImageInputBinding>,
    pub(crate) media: Option<ImageInputBinding>,
    pub(crate) denoising_strength: Option<ImageInputBinding>,
    pub(crate) application_inputs: std::collections::BTreeMap<String, ImageInputBinding>,
    pub(crate) samplers: Vec<String>,
}

impl ImagePipelineSpec {
    pub(crate) fn from_pipeline(spec: &PipelineSpec) -> Option<Self> {
        let image_output = spec
            .workflow
            .outputs
            .values()
            .find(|output| output.role == WorkflowOutputRole::Image)?;
        Some(Self {
            output_value_range: image_output.value_range?,
            prompt_tokens: binding(spec, RuntimeInputRole::PromptTokens),
            negative_prompt_tokens: binding(spec, RuntimeInputRole::NegativePromptTokens),
            seed: binding(spec, RuntimeInputRole::Seed),
            steps: binding(spec, RuntimeInputRole::MaxIterations),
            guidance_scale: binding(spec, RuntimeInputRole::GuidanceScale),
            width: binding(spec, RuntimeInputRole::Width),
            height: binding(spec, RuntimeInputRole::Height),
            media: binding(spec, RuntimeInputRole::Media)
                .filter(|binding| binding.contract.dtype == "uint8" && binding.contract.rank == 1),
            denoising_strength: binding(spec, RuntimeInputRole::DenoisingStrength),
            application_inputs: spec
                .workflow
                .inputs
                .values()
                .filter_map(|input| match (&input.source, &input.role) {
                    (WorkflowInputSource::Application { name }, SemanticInputRole::Opaque) => {
                        Some((
                            name.clone(),
                            ImageInputBinding {
                                name: name.clone(),
                                contract: input.contract.clone(),
                                default: input.default.clone(),
                                required: input.required,
                            },
                        ))
                    }
                    _ => None,
                })
                .collect(),
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

    pub(crate) fn warmup_request(
        &self,
        tokenizer: &Tokenizer,
        max_context: Option<usize>,
    ) -> anyhow::Result<ImageExecutionRequest> {
        if self.media.as_ref().is_some_and(|binding| binding.required) {
            anyhow::bail!(
                "image warmup cannot synthesize a required media input; the package must provide a canonical default or a text-to-image path"
            );
        }
        if self
            .denoising_strength
            .as_ref()
            .is_some_and(|binding| binding.required)
        {
            anyhow::bail!(
                "image warmup cannot bind required denoising_strength without a source image"
            );
        }

        let prompt_binding = self
            .prompt_tokens
            .as_ref()
            .context("image warmup requires a prompt_tokens runtime role")?;
        let mut prompt = tokenizer
            .encode("warmup")
            .context("failed to tokenize image warmup prompt")?;
        let mut negative = self
            .negative_prompt_tokens
            .as_ref()
            .map(|binding| {
                tokenizer
                    .encode("")
                    .context("failed to tokenize image warmup negative prompt")
                    .map(|tokens| (binding, tokens))
            })
            .transpose()?;
        if let Some((_, negative_tokens)) = negative.as_mut() {
            let target = prompt.len().max(negative_tokens.len());
            let pad = ["<pad>", "[PAD]", "<|pad|>"]
                .into_iter()
                .find_map(|token| tokenizer.token_id(token))
                .unwrap_or(0);
            prompt.resize(target, pad);
            negative_tokens.resize(target, pad);
        }

        let seed = Self::default_i64(self.seed.as_ref()).unwrap_or(0).max(0);
        let steps = Self::default_i64(self.steps.as_ref()).unwrap_or(1).max(1);
        let mut inputs = vec![(
            prompt_binding.name.clone(),
            token_input(prompt_binding, &prompt)?,
        )];
        if let Some((binding, tokens)) = negative {
            inputs.push((binding.name.clone(), token_input(binding, &tokens)?));
        }
        if let Some(binding) = &self.seed {
            inputs.push((binding.name.clone(), scalar_i64(binding, seed)?));
        }
        if let Some(binding) = &self.steps {
            inputs.push((binding.name.clone(), scalar_i64(binding, steps)?));
        }
        push_optional_default_f32(&mut inputs, self.guidance_scale.as_ref())?;
        push_optional_default_i64(&mut inputs, self.width.as_ref())?;
        push_optional_default_i64(&mut inputs, self.height.as_ref())?;

        Ok(ImageExecutionRequest {
            request: GenerateRequest {
                prompt: GeneratePrompt::TokenIds(prompt),
                options: GenerateOptions {
                    max_new_tokens: usize::try_from(steps)?,
                    max_context,
                    seed: Some(seed as u64),
                    ..GenerateOptions::default()
                },
            },
            inputs,
        })
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
            required: input.required,
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
    I64 {
        values: Vec<i64>,
        shape: Vec<i64>,
    },
    F32 {
        values: Vec<f32>,
        shape: Vec<i64>,
    },
    Bytes(Vec<u8>),
    Raw {
        bytes: Vec<u8>,
        shape: Vec<i64>,
        dtype: DataType,
    },
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
            Self::Raw {
                bytes,
                shape,
                dtype,
            } => Value::from_raw_bytes(bytes, &shape, dtype).map_err(Into::into),
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

pub(crate) fn token_input(
    binding: &ImageInputBinding,
    tokens: &[u32],
) -> anyhow::Result<ImageInputValue> {
    let values = tokens
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    let shape = match binding.contract.rank {
        1 => vec![values.len() as i64],
        2 => vec![1, values.len() as i64],
        rank => anyhow::bail!("prompt token binding must have rank 1 or 2, got rank {rank}"),
    };
    Ok(ImageInputValue::I64 { values, shape })
}

fn push_optional_default_i64(
    inputs: &mut Vec<(String, ImageInputValue)>,
    binding: Option<&ImageInputBinding>,
) -> anyhow::Result<()> {
    let Some(binding) = binding else {
        return Ok(());
    };
    match ImagePipelineSpec::default_i64(Some(binding)) {
        Some(value) => inputs.push((binding.name.clone(), scalar_i64(binding, value)?)),
        None if binding.required => anyhow::bail!(
            "required image warmup input '{}' has no integer default",
            binding.name
        ),
        None => {}
    }
    Ok(())
}

fn push_optional_default_f32(
    inputs: &mut Vec<(String, ImageInputValue)>,
    binding: Option<&ImageInputBinding>,
) -> anyhow::Result<()> {
    let Some(binding) = binding else {
        return Ok(());
    };
    match ImagePipelineSpec::default_f32(Some(binding)) {
        Some(value) => inputs.push((binding.name.clone(), scalar_f32(binding, value)?)),
        None if binding.required => anyhow::bail!(
            "required image warmup input '{}' has no numeric default",
            binding.name
        ),
        None => {}
    }
    Ok(())
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
