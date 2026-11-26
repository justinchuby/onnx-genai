use onnx_genai_metadata::AdapterArtifact;

/// Admitted saved factor storage; all portable arithmetic remains F32.
#[derive(Clone, Copy)]
pub(super) enum FactorStorage {
    F32,
    F16,
    Bf16,
}
impl FactorStorage {
    pub(super) fn declared(artifact: &AdapterArtifact) -> anyhow::Result<Self> {
        match artifact.dtype.as_str() {
            "float32" | "fp32" => Ok(Self::F32),
            "float16" | "fp16" => Ok(Self::F16),
            "bfloat16" | "bf16" => Ok(Self::Bf16),
            dtype => anyhow::bail!(
                "adapter {}@{} storage dtype {dtype:?} is unsupported; declare F32, F16 or BF16 ordinary linear factors",
                artifact.identity,
                artifact.version
            ),
        }
    }
    pub(super) fn dtype(self) -> onnx_runtime_ir::DataType {
        match self {
            Self::F32 => onnx_runtime_ir::DataType::Float32,
            Self::F16 => onnx_runtime_ir::DataType::Float16,
            Self::Bf16 => onnx_runtime_ir::DataType::BFloat16,
        }
    }
    pub(super) fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
    pub(super) fn decode(self, bytes: &[u8]) -> f32 {
        match self {
            Self::F32 => f32::from_le_bytes(bytes.try_into().expect("validated F32 element width")),
            Self::F16 => half::f16::from_bits(u16::from_le_bytes(
                bytes.try_into().expect("validated F16 element width"),
            ))
            .to_f32(),
            Self::Bf16 => half::bf16::from_bits(u16::from_le_bytes(
                bytes.try_into().expect("validated BF16 element width"),
            ))
            .to_f32(),
        }
    }
}
