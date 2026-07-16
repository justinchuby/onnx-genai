//! Correctness-first `com.microsoft::MatMulNBits` for f32 activations and
//! block-quantized int4 weights.
//!
//! ORT stores `B` as `[N, ceil(K / block_size), block_size / 2]`. Within each
//! block, the earlier K element occupies the low nibble. For M=1 decode,
//! constant quantized weights are dequantized once to output-major `[N, K]` f32
//! and reused by a bounded-parallel GEMV. Other shapes use the shared CPU GEMM,
//! including its oneDNN backend.

use std::sync::OnceLock;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};
use rayon::prelude::*;

use super::matmul::gemm;
use super::{check_arity, to_dense_bytes, to_dense_f32, to_dense_i64, write_dense_f32};
use crate::strided::numel;

pub struct MatMulNBitsKernel {
    k: usize,
    n: usize,
    block_size: usize,
    constant_inputs: [bool; 5],
    weight_nk: OnceLock<Vec<f32>>,
}

pub struct MatMulNBitsFactory;

impl KernelFactory for MatMulNBitsFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let bits = optional_int_attr(node, "bits")?.unwrap_or(4);
        if bits != 4 {
            return Err(error(format!(
                "only bits=4 is supported in the CPU Phase-1 kernel, got {bits}"
            )));
        }
        let weight_prepacked = optional_int_attr(node, "weight_prepacked")?.unwrap_or(0);
        if weight_prepacked != 0 {
            return Err(error(format!(
                "weight_prepacked={weight_prepacked} is unsupported: CPU only supports the standard (non-prepacked) layout"
            )));
        }
        let block_size = required_positive_attr(node, "block_size")?;
        if block_size < 16 || !block_size.is_power_of_two() {
            return Err(error(format!(
                "block_size must be a power of two and at least 16, got {block_size}"
            )));
        }

        // accuracy_level controls optimized internal compute in ORT. This
        // correctness path always accumulates in f32, so every value is safe to
        // ignore without changing the mathematical result.
        let _accuracy_level = node
            .attr("accuracy_level")
            .and_then(|value| value.as_int())
            .unwrap_or(0);

        Ok(Box::new(MatMulNBitsKernel {
            k,
            n,
            block_size,
            constant_inputs: [false; 5],
            weight_nk: OnceLock::new(),
        }))
    }
}

impl Kernel for MatMulNBitsKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("MatMulNBits", inputs, outputs, 3, 6, 1)?;
        require_dtype("A", inputs[0].dtype, DataType::Float32)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("scales", inputs[2].dtype, DataType::Float32)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size / 2;
        require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        if let Some(zp) = zero_points {
            require_dtype("zero_points", zp.dtype, DataType::Uint8)?;
            let zp_blob_size = k_blocks.div_ceil(2);
            require_flat_or_matrix_shape("zero_points", zp.shape, self.n, zp_blob_size)?;
        }

        let group_indices = optional_input(inputs, 4);
        if let Some(g_idx) = group_indices {
            require_dtype("g_idx", g_idx.dtype, DataType::Int32)?;
            let padded_k = k_blocks * self.block_size;
            if g_idx.shape != [self.k] && g_idx.shape != [padded_k] {
                return Err(error(format!(
                    "g_idx must have shape [{}] or [{padded_k}], got {:?}",
                    self.k, g_idx.shape
                )));
            }
        }

        let bias = if let Some(bias) = optional_input(inputs, 5) {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
            Some(to_dense_f32(bias)?)
        } else {
            None
        };

        let can_prepack = self.constant_inputs[1]
            && self.constant_inputs[2]
            && zero_points.is_none_or(|_| self.constant_inputs[3])
            && group_indices.is_none_or(|_| self.constant_inputs[4]);
        let activations = to_dense_f32(&inputs[0])?;
        let m = numel(&a_shape[..a_shape.len() - 1]);
        let mut result = vec![0.0f32; m * self.n];
        if can_prepack && m == 1 {
            let weight_nk = if let Some(weight) = self.weight_nk.get() {
                weight
            } else {
                let weight = self.dequantize_weight(
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    group_indices,
                    WeightLayout::Nk,
                )?;
                let _ = self.weight_nk.set(weight);
                self.weight_nk
                    .get()
                    .expect("constant MatMulNBits prepack was just initialized")
            };
            gemv_nk(&activations, weight_nk, &mut result, self.k, self.n);
        } else {
            let weight_kn = self.dequantize_weight(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                WeightLayout::Kn,
            )?;
            gemm(&activations, &weight_kn, &mut result, m, self.k, self.n)?;
        }
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

impl MatMulNBitsKernel {
    fn dequantize_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        layout: WeightLayout,
    ) -> Result<Vec<f32>> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        let group_indices = group_indices.map(to_dense_i64).transpose()?;
        let k_blocks = self.k.div_ceil(self.block_size);
        if let Some(indices) = &group_indices {
            for (index, &group) in indices.iter().enumerate() {
                if group < 0 || group as usize >= k_blocks {
                    return Err(error(format!(
                        "g_idx[{index}]={group} is outside 0..{k_blocks}"
                    )));
                }
            }
        }

        let blob_size = self.block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);
        let mut weight_kn = vec![0.0f32; self.k * self.n];
        for output in 0..self.n {
            for depth in 0..self.k {
                let block = depth / self.block_size;
                let within_block = depth % self.block_size;
                let byte = packed[(output * k_blocks + block) * blob_size + within_block / 2];
                let quantized = if within_block.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let group = group_indices
                    .as_ref()
                    .map_or(block, |indices| indices[depth] as usize);
                let zero_point = packed_zero_points.as_ref().map_or(8, |points| {
                    let byte = points[output * zp_row_bytes + group / 2];
                    if group.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }
                });
                let index = match layout {
                    WeightLayout::Kn => depth * self.n + output,
                    WeightLayout::Nk => output * self.k + depth,
                };
                weight_kn[index] =
                    (quantized as f32 - zero_point as f32) * scales[output * k_blocks + group];
            }
        }
        Ok(weight_kn)
    }
}

#[derive(Clone, Copy)]
enum WeightLayout {
    Kn,
    Nk,
}

fn gemv_nk(activation: &[f32], weight_nk: &[f32], result: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight_nk.len(), n * k);
    debug_assert_eq!(result.len(), n);
    let min_outputs_per_task = n.div_ceil(8);
    result
        .par_iter_mut()
        .with_min_len(min_outputs_per_task)
        .zip(
            weight_nk
                .par_chunks_exact(k)
                .with_min_len(min_outputs_per_task),
        )
        .for_each(|(output, weight)| {
            *output = activation.iter().zip(weight).map(|(&a, &b)| a * b).sum();
        });
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
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

fn require_flat_or_matrix_shape(
    name: &str,
    got: &[usize],
    rows: usize,
    columns: usize,
) -> Result<()> {
    if got != [rows * columns] && got != [rows, columns] {
        return Err(error(format!(
            "{name} must have shape [{}] or [{rows}, {columns}], got {got:?}",
            rows * columns
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("MatMulNBits: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use crate::CpuExecutionProvider;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{static_shape, Attribute, Graph, NodeId};
    use onnx_runtime_loader::Model;

    fn model_node(
        a_shape: &[usize],
        b_shape: &[usize],
        scales_shape: &[usize],
        zero_points_shape: Option<&[usize]>,
        output_shape: &[usize],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let mut inputs = Vec::new();
        for (name, dtype, shape) in [
            ("A", DataType::Float32, a_shape),
            ("B", DataType::Uint8, b_shape),
            ("scales", DataType::Float32, scales_shape),
        ] {
            let value = graph.create_named_value(name, dtype, static_shape(shape.iter().copied()));
            graph.add_input(value);
            inputs.push(Some(value));
        }
        if let Some(shape) = zero_points_shape {
            let value = graph.create_named_value(
                "zero_points",
                DataType::Uint8,
                static_shape(shape.iter().copied()),
            );
            graph.add_input(value);
            inputs.push(Some(value));
        }
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), "MatMulNBits", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(n as i64));
        node.attributes.insert("bits".into(), Attribute::Int(4));
        node.attributes
            .insert("block_size".into(), Attribute::Int(block_size as i64));
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn test_kernel(k: usize, n: usize, block_size: usize) -> MatMulNBitsKernel {
        MatMulNBitsKernel {
            k,
            n,
            block_size,
            constant_inputs: [false; 5],
            weight_nk: OnceLock::new(),
        }
    }

    fn quantize(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let mut packed = vec![0u8; n * blocks * blob];
        let mut scales = vec![0.0f32; n * blocks];
        let mut zps = vec![0u8; n * blocks.div_ceil(2)];
        let mut dequantized = vec![0.0f32; n * k];
        for row in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[row * k + start..row * k + end];
                let (scale, zp) = if asymmetric {
                    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = ((max - min) / 15.0).max(1e-6);
                    (scale, (-min / scale).round().clamp(0.0, 15.0) as u8)
                } else {
                    let max_abs = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
                    ((max_abs / 7.0).max(1e-6), 8)
                };
                scales[row * blocks + block] = scale;
                if asymmetric {
                    let byte = &mut zps[row * blocks.div_ceil(2) + block / 2];
                    *byte |= zp << (4 * (block % 2));
                }
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + zp as f32).round().clamp(0.0, 15.0) as u8;
                    packed[(row * blocks + block) * blob + offset / 2] |= q << (4 * (offset % 2));
                    dequantized[row * k + start + offset] = (q as f32 - zp as f32) * scale;
                }
            }
        }
        (packed, scales, asymmetric.then_some(zps), dequantized)
    }

    fn reference(a: &[f32], weights_nk: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                for depth in 0..k {
                    output[row * n + column] += a[row * k + depth] * weights_nk[column * k + depth];
                }
            }
        }
        output
    }

    fn dequantize_reference(
        packed: &[u8],
        scales: &[f32],
        zero_points: Option<&[u8]>,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let blocks = k.div_ceil(block_size);
        let blob_size = block_size / 2;
        let zp_row_bytes = blocks.div_ceil(2);
        let mut weights = vec![0.0; n * k];
        for output in 0..n {
            for depth in 0..k {
                let block = depth / block_size;
                let within_block = depth % block_size;
                let byte = packed[(output * blocks + block) * blob_size + within_block / 2];
                let q = if within_block.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let zero_point = zero_points.map_or(8, |points| {
                    let byte = points[output * zp_row_bytes + block / 2];
                    if block.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }
                });
                weights[output * k + depth] =
                    (q as f32 - zero_point as f32) * scales[output * blocks + block];
            }
        }
        weights
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn matmulnbits_symmetric_block32_matches_independent_dequantization() {
        let (m, k, n, block_size) = (3, 64, 8, 32);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) / 11.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 9.0)
            .collect();
        let (packed, scales, _, dequantized) = quantize(&weights, n, k, block_size, false);
        let (graph, node) = model_node(
            &[m, k],
            &[n, 2, 16],
            &[n, 2],
            None,
            &[m, n],
            k,
            n,
            block_size,
        );
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[m, k], &a);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &reference(&a.to_f32(), &dequantized, m, k, n));
    }

    #[test]
    fn matmulnbits_asymmetric_block16_batched_non_square() {
        let (m, k, n, block_size) = (6, 48, 5, 16);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 23) as f32 - 5.0) / 8.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 37) as f32 - 9.0) / 10.0)
            .collect();
        let (packed, scales, zero_points, dequantized) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.unwrap();
        let (graph, node) = model_node(
            &[2, 3, k],
            &[n, 3, 8],
            &[n * 3],
            Some(&[n, 2]),
            &[2, 3, n],
            k,
            n,
            block_size,
        );
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[2, 3, k], &a);
        let b = Owned::u8(&[n, 3, 8], &packed);
        let scales = Owned::f32(&[n * 3], &scales);
        let zero_points = Owned::u8(&[n, 2], &zero_points);
        let mut y = Owned::zeros_f32(&[2, 3, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zero_points.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(&y.to_f32(), &reference(&a.to_f32(), &dequantized, m, k, n));
    }

    #[test]
    fn matmulnbits_prepacked_m1_block32_symmetric_reuses_weight_for_new_activations() {
        let (k, n, block_size) = (35, 7, 32);
        let a1_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let a2_values: Vec<f32> = a1_values
            .iter()
            .enumerate()
            .map(|(i, &value)| value * -0.5 + i as f32 / 17.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true]);

        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let a1 = Owned::f32(&[1, k], &a1_values);
        let mut y1 = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a1.view(), b.view(), scales.view()], &mut [y1.view_mut()])
            .unwrap();
        assert_close(&y1.to_f32(), &reference(&a1_values, &dequantized, 1, k, n));

        let cached_weight = kernel
            .weight_nk
            .get()
            .expect("M=1 constant B must populate the prepacked weight cache")
            .as_ptr();
        let a2 = Owned::f32(&[1, k], &a2_values);
        let mut y2 = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a2.view(), b.view(), scales.view()], &mut [y2.view_mut()])
            .unwrap();
        assert_eq!(kernel.weight_nk.get().unwrap().as_ptr(), cached_weight);
        assert_close(&y2.to_f32(), &reference(&a2_values, &dequantized, 1, k, n));
        assert_ne!(y1.to_f32(), y2.to_f32());
    }

    #[test]
    fn matmulnbits_prepacked_m1_block128_explicit_zp_partial_block_matches_reference() {
        let (k, n, block_size) = (141, 7, 128);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, zero_points, _) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.unwrap();
        let dequantized =
            dequantize_reference(&packed, &scales, Some(&zero_points), n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 2, 64], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let zero_points = Owned::u8(&[n, 1], &zero_points);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zero_points.view()],
                &mut [y.view_mut()],
            )
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        assert!(
            kernel.weight_nk.get().is_some(),
            "M=1 constant B/scales/zero-points must take the prepacked GEMV path"
        );
    }

    #[test]
    fn matmulnbits_m1_dynamic_b_falls_back_without_populating_prepack_cache() {
        let (k, n, block_size) = (35, 5, 32);
        let a_values: Vec<f32> = (0..k).map(|i| ((i * 5 % 29) as f32 - 14.0) / 9.0).collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 10.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, false, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        assert!(
            kernel.weight_nk.get().is_none(),
            "dynamic B must use the fallback rather than populate the prepack cache"
        );
    }

    #[test]
    fn matmulnbits_unpacks_low_nibble_before_high_nibble() {
        let k = 16;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 16);
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![0.0; k];
        activation[0] = 1.0;
        activation[1] = 10.0;
        let mut packed = vec![0x88; 8];
        packed[0] = 0xe1;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![53.0]); // (1-8)*1 + (14-8)*10
    }

    #[test]
    fn matmulnbits_honors_non_contiguous_group_indices() {
        let k = 32;
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let a_value = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
        let b_value = graph.create_named_value("B", DataType::Uint8, static_shape([1, 2, 8]));
        let scales_value =
            graph.create_named_value("scales", DataType::Float32, static_shape([1, 2]));
        let g_idx_value = graph.create_named_value("g_idx", DataType::Int32, static_shape([k]));
        for value in [a_value, b_value, scales_value, g_idx_value] {
            graph.add_input(value);
        }
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1, 1]));
        let mut node = Node::new(
            NodeId(0),
            "MatMulNBits",
            vec![
                Some(a_value),
                Some(b_value),
                Some(scales_value),
                None,
                Some(g_idx_value),
            ],
            vec![output],
        );
        node.domain = "com.microsoft".into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(1));
        node.attributes.insert("bits".into(), Attribute::Int(4));
        node.attributes
            .insert("block_size".into(), Attribute::Int(16));
        let node = graph.insert_node(node);
        graph.add_output(output);

        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![1.0; k];
        activation[16..].fill(2.0);
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 2, 8], &vec![0x99; 16]);
        let scales = Owned::f32(&[1, 2], &[1.0, 2.0]);
        let groups: Vec<i32> = (0..k).map(|i| if i < 16 { 1 } else { 0 }).collect();
        let groups = Owned::i32(&[k], &groups);
        let absent_zp = TensorView::absent(DataType::Uint8);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), absent_zp, groups.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_eq!(y.to_f32(), vec![64.0]);
    }

    #[test]
    fn matmulnbits_rejects_non_int4_factory_configuration() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 8], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(8));
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("bits=8 must be rejected");
        assert!(format!("{error}").contains("only bits=4"));
    }

    #[test]
    fn matmulnbits_defaults_missing_bits_to_int4() {
        let k = 16;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 16);
        let mut graph = graph;
        graph.node_mut(node).attributes.remove("bits");
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("missing bits must default to 4");
        let mut activation = vec![0.0; k];
        activation[0] = 1.0;
        activation[1] = 10.0;
        let mut packed = vec![0x88; 8];
        packed[0] = 0xe1;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![53.0]);
    }

    #[test]
    fn matmulnbits_rejects_prepacked_weight_layout() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 8], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("weight_prepacked".into(), Attribute::Int(1));
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("prepacked weights must be rejected");
        let message = format!("{error}");
        assert!(message.contains("weight_prepacked=1"));
        assert!(message.contains("standard (non-prepacked) layout"));
    }
}
