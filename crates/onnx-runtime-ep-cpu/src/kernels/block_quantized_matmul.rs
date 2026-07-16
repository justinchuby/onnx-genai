//! Correctness-first `BlockQuantizedMatMul` for native GGUF block formats.
//!
//! The packed weight tensor keeps llama.cpp's serialized block layout. MXFP4
//! decoding follows OCP MX E2M1/E8M0 and llama.cpp's `block_mxfp4`; IQ4_NL
//! follows llama.cpp's `block_iq4_nl` and audited 16-entry codebook.

use std::sync::OnceLock;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::matmul::gemm;
use super::{check_arity, to_dense_bytes, to_dense_f32, write_dense_f32};
use crate::strided::numel;

const OP: &str = "BlockQuantizedMatMul";
const DOMAIN: &str = "com.github.onnxruntime.genai";
const LAYOUT_VERSION: i64 = 1;

const MXFP4_QK: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 17;
const IQ4_NL_QK: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;

// OCP E2M1 values, doubled to pair with llama.cpp's half-scale E8M0 decode.
const E2M1_DOUBLED: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// llama.cpp commit b15ca938, ggml-common.h::kvalues_iq4nl.
const IQ4_NL_CODEBOOK: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockFormat {
    Mxfp4,
    Iq4Nl,
}

impl BlockFormat {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "mxfp4" => Ok(Self::Mxfp4),
            "iq4_nl" => Ok(Self::Iq4Nl),
            "iq1_s" | "iq1_m" | "iq2_xxs" | "iq2_xs" | "iq2_s" | "iq3_xxs" | "iq3_s" | "iq4_xs" => {
                Err(error(format!(
                    "format '{value}' is recognized but not implemented; supported formats are mxfp4 and iq4_nl"
                )))
            }
            _ => Err(error(format!(
                "unsupported format '{value}'; supported formats are mxfp4 and iq4_nl"
            ))),
        }
    }

    fn qk(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_QK,
            Self::Iq4Nl => IQ4_NL_QK,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_BLOCK_BYTES,
            Self::Iq4Nl => IQ4_NL_BLOCK_BYTES,
        }
    }

    fn decode_block(self, block: &[u8], output: &mut [f32; 32]) {
        match self {
            Self::Mxfp4 => decode_mxfp4_block(block, output),
            Self::Iq4Nl => decode_iq4_nl_block(block, output),
        }
    }
}

pub struct BlockQuantizedMatMulKernel {
    k: usize,
    n: usize,
    format: BlockFormat,
    packed_b_constant: bool,
    weight_kn: OnceLock<Vec<f32>>,
}

pub struct BlockQuantizedMatMulFactory;

impl KernelFactory for BlockQuantizedMatMulFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let layout_version = optional_int_attr(node, "block_layout_version")?.unwrap_or(1);
        if layout_version != LAYOUT_VERSION {
            return Err(error(format!(
                "block_layout_version must be {LAYOUT_VERSION}, got {layout_version}"
            )));
        }
        let format = match node.attr("format") {
            Some(attribute) => attribute
                .as_str()
                .ok_or_else(|| error("attribute 'format' must be a UTF-8 string"))
                .and_then(BlockFormat::parse)?,
            None => return Err(error("missing required string attribute 'format'")),
        };

        Ok(Box::new(BlockQuantizedMatMulKernel {
            k,
            n,
            format,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        }))
    }
}

impl Kernel for BlockQuantizedMatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.packed_b_constant = constant_inputs.get(1).copied().unwrap_or(false);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 2, 3, 1)?;
        require_dtype("A", inputs[0].dtype, DataType::Float32)?;
        require_dtype("packed_B", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        require_shape("Y", outputs[0].shape, &expected_output_shape)?;

        let blocks = self.k.div_ceil(self.format.qk());
        require_shape(
            "packed_B",
            inputs[1].shape,
            &[self.n, blocks, self.format.block_bytes()],
        )?;

        let bias = if let Some(bias) = inputs.get(2).filter(|input| !input.is_absent()) {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
            Some(to_dense_f32(bias)?)
        } else {
            None
        };

        let activations = to_dense_f32(&inputs[0])?;
        let owned_weight;
        let weight_kn = if self.packed_b_constant {
            if let Some(weight) = self.weight_kn.get() {
                weight
            } else {
                let weight = self.dequantize_weight_kn(&inputs[1])?;
                let _ = self.weight_kn.set(weight);
                self.weight_kn
                    .get()
                    .expect("constant block-quantized weight was just initialized")
            }
        } else {
            owned_weight = self.dequantize_weight_kn(&inputs[1])?;
            &owned_weight
        };

        let m = numel(&a_shape[..a_shape.len() - 1]);
        let result_elements = m
            .checked_mul(self.n)
            .ok_or_else(|| error("Y element count overflow"))?;
        let mut result = vec![0.0f32; result_elements];
        gemm(&activations, weight_kn, &mut result, m, self.k, self.n)?;
        if let Some(bias) = bias {
            for row in result.chunks_exact_mut(self.n) {
                for (value, bias) in row.iter_mut().zip(&bias) {
                    *value += bias;
                }
            }
        }
        write_dense_f32(&mut outputs[0], &result)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl BlockQuantizedMatMulKernel {
    fn dequantize_weight_kn(&self, packed: &TensorView) -> Result<Vec<f32>> {
        let packed = to_dense_bytes(packed)?;
        let qk = self.format.qk();
        let block_bytes = self.format.block_bytes();
        let blocks = self.k.div_ceil(qk);
        let expected_bytes = self
            .n
            .checked_mul(blocks)
            .and_then(|value| value.checked_mul(block_bytes))
            .ok_or_else(|| error("packed_B byte count overflow"))?;
        if packed.len() != expected_bytes {
            return Err(error(format!(
                "packed_B must contain exactly {expected_bytes} bytes, got {}",
                packed.len()
            )));
        }

        let weight_elements = self
            .k
            .checked_mul(self.n)
            .ok_or_else(|| error("dequantized weight element count overflow"))?;
        let mut weight_kn = vec![0.0f32; weight_elements];
        let mut decoded = [0.0f32; 32];
        for output in 0..self.n {
            for block_index in 0..blocks {
                let packed_start = (output * blocks + block_index) * block_bytes;
                self.format.decode_block(
                    &packed[packed_start..packed_start + block_bytes],
                    &mut decoded,
                );
                let k_start = block_index * qk;
                let valid = (self.k - k_start).min(qk);
                for (offset, value) in decoded[..valid].iter().copied().enumerate() {
                    weight_kn[(k_start + offset) * self.n + output] = value;
                }
            }
        }
        Ok(weight_kn)
    }
}

fn decode_mxfp4_block(block: &[u8], output: &mut [f32; 32]) {
    debug_assert_eq!(block.len(), MXFP4_BLOCK_BYTES);
    let half_scale = e8m0_half_scale(block[0]);
    for j in 0..16 {
        let packed = block[1 + j];
        output[j] = E2M1_DOUBLED[(packed & 0x0f) as usize] as f32 * half_scale;
        output[j + 16] = E2M1_DOUBLED[(packed >> 4) as usize] as f32 * half_scale;
    }
}

fn e8m0_half_scale(exponent: u8) -> f32 {
    match exponent {
        // OCP E8M0 reserves 0xff for NaN. llama.cpp does not emit it.
        0xff => f32::NAN,
        // Exact subnormal representations of 2^-128 and 2^-127.
        0 => f32::from_bits(0x0020_0000),
        1 => f32::from_bits(0x0040_0000),
        // Half of 2^(e-127) is 2^(e-128), encoded with f32 exponent e-1.
        _ => f32::from_bits((u32::from(exponent) - 1) << 23),
    }
}

fn decode_iq4_nl_block(block: &[u8], output: &mut [f32; 32]) {
    debug_assert_eq!(block.len(), IQ4_NL_BLOCK_BYTES);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..16 {
        let packed = block[2 + j];
        output[j] = scale * IQ4_NL_CODEBOOK[(packed & 0x0f) as usize] as f32;
        output[j + 16] = scale * IQ4_NL_CODEBOOK[(packed >> 4) as usize] as f32;
    }
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = optional_int_attr(node, name)?
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    match node.attr(name) {
        Some(attribute) => attribute
            .as_int()
            .map(Some)
            .ok_or_else(|| error(format!("attribute '{name}' must be an integer"))),
        None => Ok(None),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{DOMAIN}::{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    fn model_node(
        format: &str,
        a_shape: &[usize],
        b_shape: &[usize],
        output_shape: &[usize],
        k: usize,
        n: usize,
        with_bias: bool,
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(DOMAIN.into(), 1);
        let a = graph.create_named_value(
            "A",
            DataType::Float32,
            static_shape(a_shape.iter().copied()),
        );
        graph.add_input(a);
        let packed_b = graph.create_named_value(
            "packed_B",
            DataType::Uint8,
            static_shape(b_shape.iter().copied()),
        );
        graph.add_input(packed_b);
        let mut inputs = vec![Some(a), Some(packed_b)];
        if with_bias {
            let bias = graph.create_named_value("bias", DataType::Float32, static_shape([n]));
            graph.add_input(bias);
            inputs.push(Some(bias));
        }
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), OP, inputs, vec![output]);
        node.domain = DOMAIN.into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(n as i64));
        node.attributes.insert(
            "format".into(),
            Attribute::String(format.as_bytes().to_vec()),
        );
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn kernel(graph: &Graph, node: NodeId) -> Box<dyn Kernel> {
        let model = Model::new(graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("CPU EP must register BlockQuantizedMatMul")
    }

    #[test]
    fn mxfp4_known_block_matches_ocp_e2m1_and_llama_layout() {
        let mut packed = vec![127u8];
        packed.extend((0u8..16).map(|code| code | (code << 4)));
        let view = Owned::u8(&[1, 1, 17], &packed);
        let kernel = BlockQuantizedMatMulKernel {
            k: 32,
            n: 1,
            format: BlockFormat::Mxfp4,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = kernel.dequantize_weight_kn(&view.view()).unwrap();
        let values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let mut expected = Vec::with_capacity(32);
        expected.extend(values);
        expected.extend(values.map(|value| -value));
        expected.extend(values);
        expected.extend(values.map(|value| -value));
        assert_eq!(actual, expected);
    }

    #[test]
    fn e8m0_decode_covers_subnormal_extremes_and_nan() {
        assert_eq!((e8m0_half_scale(0) * 2.0).to_bits(), 0x0040_0000);
        assert_eq!((e8m0_half_scale(1) * 2.0).to_bits(), 0x0080_0000);
        assert_eq!(e8m0_half_scale(127), 0.5);
        assert_eq!(e8m0_half_scale(128), 1.0);
        assert_eq!((e8m0_half_scale(254) * 2.0).to_bits(), 0x7f00_0000);
        assert!(e8m0_half_scale(255).is_nan());
    }

    #[test]
    fn mxfp4_batched_matmul_with_partial_block_and_bias_matches_reference() {
        let (m, k, n): (usize, usize, usize) = (2, 45, 2);
        let blocks = k.div_ceil(32);
        let mut packed = vec![0u8; n * blocks * MXFP4_BLOCK_BYTES];
        let mut weight_nk = vec![0.0f32; n * k];
        for output in 0..n {
            for block in 0..blocks {
                let start = (output * blocks + block) * MXFP4_BLOCK_BYTES;
                packed[start] = 127 + output as u8;
                for j in 0..16 {
                    let low = ((j + block + output) % 16) as u8;
                    let high = ((15 + output - (j % 2)) % 16) as u8;
                    packed[start + 1 + j] = low | (high << 4);
                }
                let mut decoded = [0.0; 32];
                decode_mxfp4_block(&packed[start..start + MXFP4_BLOCK_BYTES], &mut decoded);
                for offset in 0..(k - block * 32).min(32) {
                    weight_nk[output * k + block * 32 + offset] = decoded[offset];
                }
            }
        }
        let activations: Vec<f32> = (0..m * k)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 8.0)
            .collect();
        let bias = [0.25, -0.5];
        let mut expected = vec![0.0; m * n];
        for row in 0..m {
            for output in 0..n {
                expected[row * n + output] = bias[output]
                    + (0..k)
                        .map(|inner| activations[row * k + inner] * weight_nk[output * k + inner])
                        .sum::<f32>();
            }
        }

        let (graph, node) = model_node("mxfp4", &[m, k], &[n, blocks, 17], &[m, n], k, n, true);
        let kernel = kernel(&graph, node);
        let a = Owned::f32(&[m, k], &activations);
        let b = Owned::u8(&[n, blocks, 17], &packed);
        let bias = Owned::f32(&[n], &bias);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        for (actual, expected) in y.to_f32().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn iq4_nl_uses_llama_codebook_and_fp16_scale() {
        let mut packed = half::f16::from_f32(0.5).to_le_bytes().to_vec();
        packed.extend((0u8..16).map(|code| code | ((15 - code) << 4)));
        let view = Owned::u8(&[1, 1, IQ4_NL_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: 32,
            n: 1,
            format: BlockFormat::Iq4Nl,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let expected: Vec<f32> = IQ4_NL_CODEBOOK
            .iter()
            .chain(IQ4_NL_CODEBOOK.iter().rev())
            .map(|value| *value as f32 * 0.5)
            .collect();
        assert_eq!(actual, expected);

        let activation: Vec<f32> = (1..=32).map(|value| value as f32 / 16.0).collect();
        let reference = activation
            .iter()
            .zip(&expected)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let (graph, node) = model_node("iq4_nl", &[1, 32], &[1, 1, 18], &[1, 1], 32, 1, false);
        let kernel = kernel(&graph, node);
        let a = Owned::f32(&[1, 32], &activation);
        let b = Owned::u8(&[1, 1, 18], &packed);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view()], &mut [y.view_mut()])
            .unwrap();
        assert!((y.to_f32()[0] - reference).abs() <= 1e-5);
    }

    #[test]
    fn incomplete_iq_format_is_rejected_at_kernel_creation() {
        let (graph, node) = model_node("iq2_xs", &[1, 256], &[1, 1, 74], &[1, 1], 256, 1, false);
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("incomplete IQ format must be rejected");
        assert!(error.to_string().contains("recognized but not implemented"));
    }
}
