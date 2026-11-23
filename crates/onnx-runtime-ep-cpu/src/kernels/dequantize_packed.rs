//! Portable standalone DequantizePacked-29. GGML-derived decoding is covered
//! by the MIT notice in onnx-runtime-quantization.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::dequantize_packed::{Format, Metadata};
use onnx_runtime_ir::{DataType, DeviceType, Node};

use super::block_quantized_matmul::BlockFormat;

pub(super) struct Factory;

struct PackedKernel(Metadata);

fn invalid(detail: String) -> EpError {
    EpError::KernelFailed(format!("DequantizePacked: {detail}"))
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(PackedKernel(
            Metadata::from_node(node).map_err(invalid)?,
        )))
    }
}

impl Kernel for PackedKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        super::check_arity("DequantizePacked", inputs, outputs, 1, 1, 1)?;
        let x = &inputs[0];
        let y = &mut outputs[0];
        let metadata = &self.0;
        metadata.validate_input(x.dtype, x.shape).map_err(invalid)?;
        metadata
            .validate_output(y.dtype, y.shape)
            .map_err(invalid)?;
        if x.device.device_type != DeviceType::Cpu || y.device.device_type != DeviceType::Cpu {
            return Err(invalid(
                "CPU kernel requires host input and output buffers".into(),
            ));
        }
        if metadata.elements() == 0 {
            return Ok(());
        }
        x.validate()?;
        y.validate()?;
        if !x.is_contiguous() || !y.is_contiguous() {
            return Err(invalid(
                "input and output must use contiguous storage".into(),
            ));
        }
        // SAFETY: CPU tensor views guarantee live nonoverlapping storage;
        // validated shapes, dtypes, and layout give these exact byte lengths.
        let src =
            unsafe { std::slice::from_raw_parts(x.data_ptr::<u8>(), metadata.packed_bytes()) };
        let dst = unsafe {
            std::slice::from_raw_parts_mut(y.data_ptr_mut::<u8>(), metadata.output_bytes())
        };
        let (width, bytes) = metadata.format().block();
        let element_bytes = metadata.dtype().byte_size();
        let mut block = [0.0; 256];
        let decode = decoder(metadata.format());
        for (src, dst) in src
            .chunks_exact(bytes)
            .zip(dst.chunks_exact_mut(width * element_bytes))
        {
            let values = &mut block[..width];
            decode(src, values);
            for (&value, output) in values.iter().zip(dst.chunks_exact_mut(element_bytes)) {
                match metadata.dtype() {
                    DataType::Float32 => output.copy_from_slice(&value.to_le_bytes()),
                    DataType::Float64 => output.copy_from_slice(&f64::from(value).to_le_bytes()),
                    DataType::Float16 => {
                        output.copy_from_slice(&half::f16::from_f32(value).to_le_bytes())
                    }
                    DataType::BFloat16 => {
                        output.copy_from_slice(&half::bf16::from_f32(value).to_le_bytes())
                    }
                    _ => unreachable!("checked output dtype"),
                }
            }
        }
        Ok(())
    }
}

fn decoder(format: Format) -> fn(&[u8], &mut [f32]) {
    match format {
        Format::Iq4Nl => BlockFormat::Iq4Nl,
        Format::Iq4Xs => BlockFormat::Iq4Xs,
        Format::Iq2Xxs => BlockFormat::Iq2Xxs,
        Format::Iq2Xs => BlockFormat::Iq2Xs,
        Format::Iq2S => BlockFormat::Iq2S,
        Format::Iq3Xxs => BlockFormat::Iq3Xxs,
        Format::Iq3S => BlockFormat::Iq3S,
        Format::Iq1S => BlockFormat::Iq1S,
        Format::Iq1M => BlockFormat::Iq1M,
        Format::Q2K => BlockFormat::Q2K,
        Format::Q3K => BlockFormat::Q3K,
        Format::Q5K => BlockFormat::Q5K,
        Format::Q6K => BlockFormat::Q6K,
        Format::Q8_0 => BlockFormat::Q8_0,
        Format::Q1_0 => return decode_lanes::<41>,
        Format::Q2_0 => return decode_lanes::<42>,
        Format::Q4_0 => return decode_lanes::<2>,
        Format::Q4_1 => return decode_lanes::<3>,
        Format::Q5_0 => return decode_lanes::<6>,
        Format::Q5_1 => return decode_lanes::<7>,
        Format::Q8_1 => return decode_lanes::<9>,
        Format::Q4K => return decode_lanes::<12>,
        Format::Q8K => return decode_lanes::<15>,
        Format::Tq1_0 => return decode_lanes::<34>,
        Format::Tq2_0 => return decode_lanes::<35>,
        Format::Mxfp4 => return decode_lanes::<39>,
        Format::Nvfp4 => return decode_lanes::<40>,
    }
    .scalar_decoder()
}

fn fp16(data: &[u8], offset: usize) -> f32 {
    half::f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32()
}

/// Plane-major field order within each byte group.
fn plane(data: &[u8], j: usize, bits: usize, group: usize) -> i32 {
    let fields = 8 / bits;
    let byte = data[j / (group * fields) * group + j % group];
    i32::from((byte >> (bits * (j / group % fields))) & ((1u16 << bits) - 1) as u8)
}

fn decode_lanes<const FORMAT: u8>(b: &[u8], values: &mut [f32]) {
    const E2M1: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    for (j, value) in values.iter_mut().enumerate() {
        *value = match FORMAT {
            2 | 3 | 6 | 7 => {
                let affine = FORMAT == 3 || FORMAT == 7;
                let five = FORMAT == 6 || FORMAT == 7;
                let h = if affine { 4 } else { 2 };
                let q = plane(&b[h + if five { 4 } else { 0 }..], j, 4, 16)
                    + if five {
                        16 * plane(&b[h..], j, 1, 1)
                    } else {
                        0
                    };
                let d = fp16(b, 0);
                if affine {
                    d * q as f32 + fp16(b, 2)
                } else {
                    d * (q - if five { 16 } else { 8 }) as f32
                }
            }
            9 => fp16(b, 0) * f32::from(b[4 + j] as i8),
            12 => {
                let s = &b[4..16];
                let k = j / 32;
                let (a, m) = if k < 4 {
                    (s[k] & 63, s[k + 4] & 63)
                } else {
                    (
                        (s[k + 4] & 15) | ((s[k - 4] >> 6) << 4),
                        (s[k + 4] >> 4) | ((s[k] >> 6) << 4),
                    )
                };
                fp16(b, 0) * f32::from(a) * plane(&b[16..], j, 4, 32) as f32
                    - fp16(b, 2) * f32::from(m)
            }
            15 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * f32::from(b[4 + j] as i8),
            34 => {
                let (start, width, offset) = if j < 160 {
                    (0, 32, j)
                } else if j < 240 {
                    (32, 16, j - 160)
                } else {
                    (48, 4, j - 240)
                };
                let factor = [1u8, 3, 9, 27, 81][offset / width];
                let wrapped = b[start + offset % width].wrapping_mul(factor);
                fp16(b, 52) * (i32::from(wrapped) * 3 / 256 - 1) as f32
            }
            35 => fp16(b, 64) * (plane(b, j, 2, 32) - 1) as f32,
            39 => {
                let e = u32::from(b[0]);
                let bits = if e < 2 {
                    0x0020_0000 << e
                } else {
                    (e - 1) << 23
                };
                f32::from_bits(bits) * f32::from(E2M1[plane(&b[1..], j, 4, 16) as usize])
            }
            40 => {
                let group = j / 16;
                let e = b[group];
                let mantissa = e & 7;
                let exponent = u32::from((e >> 3) & 15);
                let scale = if e == 0 || e == 127 {
                    0.0
                } else if exponent == 0 {
                    f32::from(mantissa) * (1.0 / 1024.0)
                } else {
                    f32::from_bits(((exponent + 119) << 23) | (u32::from(mantissa) << 20))
                };
                scale * f32::from(E2M1[plane(&b[4..], j, 4, 8) as usize])
            }
            41 => {
                let d = fp16(b, 0);
                if plane(&b[2..], j, 1, 1) == 1 { d } else { -d }
            }
            42 => fp16(b, 0) * (plane(&b[2..], j, 2, 1) - 1) as f32,
            _ => unreachable!("checked packed format"),
        };
    }
}
