//! `com.microsoft::GatherBlockQuantized` — gather rows from a block-quantized
//! `data` tensor and dequantize them on the fly.
//!
//! Faithful CPU port of ONNX Runtime's contrib kernel
//! (`contrib_ops/cpu/quantization/gather_block_quantized.cc`) for the `uint8`
//! storage type (the Qwen3.5 embedding table is stored as `uint8` with
//! `bits = 8`). It also handles the packed `bits ∈ {2, 4}` uint8 layouts ORT
//! supports, so the kernel stays general rather than pinned to one model
//! (RULES.md §2).
//!
//! ## Contract
//!
//! Inputs:
//! * `data` — `uint8` block-quantized weights. For `bits < 8` several logical
//!   elements are packed per byte, so the logical last-axis extent is
//!   `data.shape[last] * (8 / bits)`.
//! * `indices` — integer gather indices (int32/int64) along `gather_axis`.
//! * `scales` — per-block dequant scales (`f32`/`f16`), same rank as `data`;
//!   the quantize axis is divided into `ceil(dim / block_size)` blocks.
//! * `zero_points` — optional, same layout as `scales` (packed for `bits < 8`);
//!   defaults to `1 << (bits - 1)` when absent.
//!
//! Attributes: `gather_axis` (default 0), `quantize_axis` (default 1),
//! `block_size` (default 128, power of two ≥ 16), `bits` (default 4).
//!
//! For `uint8` data ORT constrains `gather_axis == 0` and
//! `quantize_axis == last axis`; we enforce the same. Output shape is
//! `indices.shape ++ data.shape[1:]` (last axis scaled by `8 / bits`), and the
//! dequantized value is `(q - zero_point) · scale` widened through `f32`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_bytes, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

pub struct GatherBlockQuantizedKernel {
    gather_axis: i64,
    quantize_axis: i64,
    block_size: i64,
    bits: i64,
}

pub struct GatherBlockQuantizedFactory;

impl KernelFactory for GatherBlockQuantizedFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let gather_axis = node
            .attr("gather_axis")
            .and_then(|a| a.as_int())
            .unwrap_or(0);
        let quantize_axis = node
            .attr("quantize_axis")
            .and_then(|a| a.as_int())
            .unwrap_or(1);
        let block_size = node
            .attr("block_size")
            .and_then(|a| a.as_int())
            .unwrap_or(128);
        let bits = node.attr("bits").and_then(|a| a.as_int()).unwrap_or(4);
        if block_size < 16 || (block_size & (block_size - 1)) != 0 {
            return Err(EpError::KernelFailed(format!(
                "GatherBlockQuantized: block_size must be a power of two >= 16, got {block_size}"
            )));
        }
        if !matches!(bits, 2 | 4 | 8) {
            return Err(EpError::KernelFailed(format!(
                "GatherBlockQuantized: only uint8 data with bits in {{2, 4, 8}} is supported, got \
                 bits={bits}"
            )));
        }
        Ok(Box::new(GatherBlockQuantizedKernel {
            gather_axis,
            quantize_axis,
            block_size,
            bits,
        }))
    }
}

/// Extract one logical `bits`-wide element from packed `uint8` storage.
#[inline]
fn extract_element(data: &[u8], data_idx: usize, bits: i64) -> i32 {
    match bits {
        8 => i32::from(data[data_idx]),
        4 => {
            let byte = data[data_idx >> 1];
            let nibble = if data_idx & 1 == 1 {
                (byte >> 4) & 0x0F
            } else {
                byte & 0x0F
            };
            i32::from(nibble)
        }
        // bits == 2
        _ => {
            let byte = data[data_idx >> 2];
            let shift = (data_idx & 3) * 2;
            i32::from((byte >> shift) & 0x03)
        }
    }
}

/// Extract one logical `bits`-wide zero point from packed `uint8` storage,
/// mirroring ORT's per-row addressing (packing is only along the quantize
/// axis, so the flat `scale_idx` is decomposed into row / within-row indices).
#[inline]
fn extract_zero_point(zp: &[u8], scale_idx: usize, scale_qaxis_dim: usize, bits: i64) -> i32 {
    match bits {
        8 => i32::from(zp[scale_idx]),
        4 => {
            let scale_row = scale_idx / scale_qaxis_dim;
            let q_in_row = scale_idx % scale_qaxis_dim;
            let packed = scale_qaxis_dim.div_ceil(2);
            let byte = zp[scale_row * packed + (q_in_row >> 1)];
            let nibble = if q_in_row & 1 == 1 {
                (byte >> 4) & 0x0F
            } else {
                byte & 0x0F
            };
            i32::from(nibble)
        }
        // bits == 2
        _ => {
            let scale_row = scale_idx / scale_qaxis_dim;
            let q_in_row = scale_idx % scale_qaxis_dim;
            let packed = scale_qaxis_dim.div_ceil(4);
            let byte = zp[scale_row * packed + (q_in_row >> 2)];
            let shift = (q_in_row & 3) * 2;
            i32::from((byte >> shift) & 0x03)
        }
    }
}

impl Kernel for GatherBlockQuantizedKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // data, indices, scales required; zero_points optional.
        check_arity("GatherBlockQuantized", inputs, outputs, 3, 4, 1)?;

        let data_shape = inputs[0].shape;
        let data_rank = data_shape.len();
        if data_rank < 2 {
            return Err(EpError::KernelFailed(format!(
                "GatherBlockQuantized: data must be rank >= 2, got shape {data_shape:?}"
            )));
        }
        let normalize_axis = |axis: i64| -> Result<usize> {
            let rank = data_rank as i64;
            let a = if axis < 0 { axis + rank } else { axis };
            if a < 0 || a >= rank {
                return Err(EpError::KernelFailed(format!(
                    "GatherBlockQuantized: axis {axis} out of range for rank {data_rank}"
                )));
            }
            Ok(a as usize)
        };
        let gather_axis = normalize_axis(self.gather_axis)?;
        let quantize_axis = normalize_axis(self.quantize_axis)?;
        // ORT constrains the uint8 storage path to these axes.
        if gather_axis != 0 || quantize_axis != data_rank - 1 {
            return Err(EpError::KernelFailed(format!(
                "GatherBlockQuantized: uint8 data requires gather_axis=0 and quantize_axis=last, \
                 got gather_axis={gather_axis}, quantize_axis={quantize_axis} (rank {data_rank})"
            )));
        }

        let components = (8 / self.bits) as usize;
        let bits = self.bits;
        let block_size = self.block_size as usize;

        // Logical geometry (ORT `Compute`): reshape data to
        // [gather_M, gather_axis_dim, gather_block] and quantize view to
        // [_, quantize_axis_dim, quantize_N].
        let gather_axis_dim = data_shape[gather_axis];
        let gather_block: usize =
            data_shape[gather_axis + 1..].iter().product::<usize>() * components;
        let gather_m: usize = data_shape[..gather_axis].iter().product();
        let quantize_axis_dim = data_shape[quantize_axis] * components;
        let quantize_n: usize = data_shape[quantize_axis + 1..].iter().product();

        let quantize_full_block = quantize_axis_dim * quantize_n;
        let scale_qaxis_dim = quantize_axis_dim.div_ceil(block_size);
        let scale_full_block = scale_qaxis_dim * quantize_n;
        let data_full_block = gather_axis_dim * gather_block;

        let indices = to_dense_i64(&inputs[1])?;
        let gather_n = indices.len();

        // Zero-copy row access: the `data` table is large (the embedding matrix
        // is hundreds of MB), so densifying it every step is prohibitive. Graph
        // initializers are contiguous, so borrow the raw buffers directly and
        // only touch the gathered rows; fall back to a dense copy for the rare
        // strided view.
        let data_owned;
        let data: &[u8] = if inputs[0].is_contiguous() {
            // SAFETY: contiguous view over a validated tensor of `numel` u8 elems.
            unsafe { std::slice::from_raw_parts(inputs[0].data_ptr::<u8>(), inputs[0].numel()) }
        } else {
            data_owned = to_dense_bytes(&inputs[0])?;
            &data_owned
        };
        let scales_f32_owned;
        let scales_widened;
        let scales: &[f32] = if inputs[2].dtype == DataType::Float32 && inputs[2].is_contiguous() {
            // SAFETY: contiguous f32 view over a validated tensor.
            scales_f32_owned = unsafe {
                std::slice::from_raw_parts(inputs[2].data_ptr::<f32>(), inputs[2].numel())
            };
            scales_f32_owned
        } else {
            scales_widened = to_dense_f32_widen("GatherBlockQuantized", &inputs[2])?;
            &scales_widened
        };
        let zp_owned;
        let zero_points: Option<&[u8]> = if inputs.len() >= 4 && !inputs[3].is_absent() {
            if inputs[3].is_contiguous() {
                // SAFETY: contiguous u8 view over a validated tensor.
                Some(unsafe {
                    std::slice::from_raw_parts(inputs[3].data_ptr::<u8>(), inputs[3].numel())
                })
            } else {
                zp_owned = to_dense_bytes(&inputs[3])?;
                Some(&zp_owned)
            }
        } else {
            None
        };
        let default_zp = 1i32 << (bits - 1);

        let mut out = vec![0.0f32; gather_m * gather_n * gather_block];

        for gather_mn_idx in 0..(gather_m * gather_n) {
            let gather_m_idx = gather_mn_idx / gather_n;
            let gather_n_idx = gather_mn_idx % gather_n;

            let raw = indices[gather_n_idx];
            let indices_val = if raw < 0 {
                raw + gather_axis_dim as i64
            } else {
                raw
            };
            if indices_val < 0 || indices_val >= gather_axis_dim as i64 {
                return Err(EpError::KernelFailed(format!(
                    "GatherBlockQuantized: index {raw} out of bounds for gather axis dim \
                     {gather_axis_dim}"
                )));
            }
            let indices_val = indices_val as usize;

            let output_idx_base = gather_mn_idx * gather_block;
            let data_idx_base = gather_m_idx * data_full_block + indices_val * gather_block;

            for i in 0..gather_block {
                let data_idx = data_idx_base + i;
                let data_val = extract_element(data, data_idx, bits);

                let x = data_idx / quantize_full_block;
                let y = (data_idx % quantize_full_block) / quantize_n;
                let z = data_idx % quantize_n;
                let scale_idx = x * scale_full_block + (y / block_size) * quantize_n + z;
                let scale_val = scales[scale_idx];

                let zp_val = match &zero_points {
                    Some(zp) => extract_zero_point(zp, scale_idx, scale_full_block, bits),
                    None => default_zp,
                };

                out[output_idx_base + i] = (data_val - zp_val) as f32 * scale_val;
            }
        }

        write_dense_f32_narrow("GatherBlockQuantized", &mut outputs[0], &out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, NodeId};

    fn node(bits: i64, block_size: i64, gather_axis: i64, quantize_axis: i64) -> Node {
        let mut node = Node::new(NodeId(0), "GatherBlockQuantized", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes
            .insert("bits".to_string(), Attribute::Int(bits));
        node.attributes
            .insert("block_size".to_string(), Attribute::Int(block_size));
        node.attributes
            .insert("gather_axis".to_string(), Attribute::Int(gather_axis));
        node.attributes
            .insert("quantize_axis".to_string(), Attribute::Int(quantize_axis));
        node
    }

    fn run(kernel: &dyn Kernel, inputs: &[&Owned], out_shape: &[usize]) -> Vec<f32> {
        let views: Vec<_> = inputs.iter().map(|o| o.view()).collect();
        let mut out = Owned::zeros_f32(out_shape);
        {
            let mut outs = [out.view_mut()];
            kernel.execute(&views, &mut outs).unwrap();
        }
        out.to_f32()
    }

    #[test]
    fn bits8_hand_computed() {
        // vocab=3, feat=16, one block (block_size=16). Gather rows 1 and 2.
        let feat = 16usize;
        let data: Vec<u8> = (0..3 * feat).map(|v| (v % 256) as u8).collect();
        let scales = [0.5f32, 0.25, 2.0];
        let zp = [8u8, 4, 128];
        let data_t = Owned::u8(&[3, feat], &data);
        let indices = Owned::i64(&[2], &[1, 2]);
        let scales_t = Owned::f32(&[3, 1], &scales);
        let zp_t = Owned::u8(&[3, 1], &zp);

        let kernel = GatherBlockQuantizedFactory
            .create(&node(8, 16, 0, 1), &[])
            .unwrap();
        let out = run(
            kernel.as_ref(),
            &[&data_t, &indices, &scales_t, &zp_t],
            &[2, feat],
        );

        let mut expected = Vec::new();
        for &idx in &[1usize, 2usize] {
            for j in 0..feat {
                let q = data[idx * feat + j] as i32;
                expected.push((q - zp[idx] as i32) as f32 * scales[idx]);
            }
        }
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, want {b}");
        }
    }

    #[test]
    fn bits8_default_zero_point() {
        // No zero_points input -> default zp = 1 << (bits-1) = 128.
        let feat = 16usize;
        let data: Vec<u8> = (0..feat).map(|v| (v * 7 % 256) as u8).collect();
        let scale = 0.75f32;
        let data_t = Owned::u8(&[1, feat], &data);
        let indices = Owned::i64(&[1], &[0]);
        let scales_t = Owned::f32(&[1, 1], &[scale]);

        let kernel = GatherBlockQuantizedFactory
            .create(&node(8, 16, 0, 1), &[])
            .unwrap();
        let out = run(kernel.as_ref(), &[&data_t, &indices, &scales_t], &[1, feat]);

        for j in 0..feat {
            let want = (data[j] as i32 - 128) as f32 * scale;
            assert!((out[j] - want).abs() < 1e-6, "got {}, want {want}", out[j]);
        }
    }

    #[test]
    fn negative_index_wraps() {
        let feat = 16usize;
        let data: Vec<u8> = (0..3 * feat).map(|v| (v % 256) as u8).collect();
        let data_t = Owned::u8(&[3, feat], &data);
        // -1 refers to the last row (index 2).
        let indices = Owned::i64(&[1], &[-1]);
        let scales_t = Owned::f32(&[3, 1], &[1.0, 1.0, 1.0]);
        let kernel = GatherBlockQuantizedFactory
            .create(&node(8, 16, 0, 1), &[])
            .unwrap();
        let out = run(kernel.as_ref(), &[&data_t, &indices, &scales_t], &[1, feat]);
        for j in 0..feat {
            let want = (data[2 * feat + j] as i32 - 128) as f32;
            assert!((out[j] - want).abs() < 1e-6);
        }
    }

    #[test]
    fn bits4_packed_hand_computed() {
        // 1 row, packed uint8 [1, 8] -> 16 logical 4-bit elements, one block.
        // zero_points packed: (scale_qaxis_dim=1) -> 1 byte, low nibble used.
        let packed: Vec<u8> = vec![0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F];
        let data_t = Owned::u8(&[1, 8], &packed);
        let indices = Owned::i64(&[1], &[0]);
        let scale = 0.5f32;
        let scales_t = Owned::f32(&[1, 1], &[scale]);
        let zp_t = Owned::u8(&[1, 1], &[0x03]); // low nibble zp = 3

        let kernel = GatherBlockQuantizedFactory
            .create(&node(4, 16, 0, 1), &[])
            .unwrap();
        let out = run(
            kernel.as_ref(),
            &[&data_t, &indices, &scales_t, &zp_t],
            &[1, 16],
        );

        for j in 0..16usize {
            let byte = packed[j >> 1];
            let nibble = if j & 1 == 1 {
                (byte >> 4) & 0x0F
            } else {
                byte & 0x0F
            } as i32;
            let want = (nibble - 3) as f32 * scale;
            assert!(
                (out[j] - want).abs() < 1e-6,
                "j={j} got {} want {want}",
                out[j]
            );
        }
    }
}
