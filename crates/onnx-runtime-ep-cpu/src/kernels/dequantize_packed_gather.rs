//! Decode only rows selected by `Gather(axis=0)` from packed GGUF storage.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::dequantize_packed::Metadata;
use onnx_runtime_ir::{DataType, DeviceType, Node, dequantize_packed_gather};

use super::dequantize_packed::decoder;
use super::to_dense_i64;

pub(super) struct Factory;

struct PackedGatherKernel {
    metadata: Metadata,
}

fn invalid(detail: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("DequantizePackedGather: {}", detail.into()))
}

fn normalize_index(raw: i64, rows: usize) -> Result<usize> {
    let index = if raw < 0 {
        i128::from(raw) + rows as i128
    } else {
        i128::from(raw)
    };
    if index < 0 || index >= rows as i128 {
        return Err(invalid(format!(
            "index {raw} is out of range for decoded axis 0 dimension {rows}"
        )));
    }
    Ok(index as usize)
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(PackedGatherKernel {
            metadata: Metadata::from_node(node).map_err(invalid)?,
        }))
    }
}

impl Kernel for PackedGatherKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let ([packed, indices], [output]) = (inputs, outputs) else {
            return Err(invalid(
                "expected packed data and indices inputs and one output",
            ));
        };
        let layout = dequantize_packed_gather::layout(
            &self.metadata,
            packed.dtype,
            packed.shape,
            indices.dtype,
            indices.shape,
            output.dtype,
            output.shape,
        )
        .map_err(invalid)?;
        if packed.device.device_type != DeviceType::Cpu
            || indices.device.device_type != DeviceType::Cpu
            || output.device.device_type != DeviceType::Cpu
        {
            return Err(invalid("CPU kernel requires host input and output buffers"));
        }
        packed.validate()?;
        output.validate()?;
        if !packed.is_contiguous() || !output.is_contiguous() {
            return Err(invalid(
                "packed input and output must use contiguous storage",
            ));
        }
        let selected = to_dense_i64(indices)?
            .into_iter()
            .map(|raw| normalize_index(raw, layout.rows))
            .collect::<Result<Vec<_>>>()?;
        if layout.lanes == 0 {
            return Ok(());
        }

        let (block_width, block_bytes) = self.metadata.format().block();
        let blocks_per_row = layout.row_elements / block_width;
        let packed_row_bytes = blocks_per_row
            .checked_mul(block_bytes)
            .ok_or_else(|| invalid("packed row byte count overflows host indexing"))?;
        let element_bytes = self.metadata.dtype().byte_size();
        let output_row_bytes = layout
            .row_elements
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid("decoded row byte count overflows host indexing"))?;
        // SAFETY: validated contiguous views have the exact byte lengths
        // established by Metadata and the output Gather geometry.
        let source = unsafe {
            std::slice::from_raw_parts(packed.data_ptr::<u8>(), self.metadata.packed_bytes())
        };
        let destination = unsafe {
            std::slice::from_raw_parts_mut(
                output.data_ptr_mut::<u8>(),
                self.metadata.dtype().storage_bytes(layout.lanes),
            )
        };
        let decode = decoder(self.metadata.format());
        let mut block = [0.0f32; 256];
        for (output_row, &source_row) in selected.iter().enumerate() {
            let source_start = source_row
                .checked_mul(packed_row_bytes)
                .ok_or_else(|| invalid("packed source row offset overflows host indexing"))?;
            let output_start = output_row
                .checked_mul(output_row_bytes)
                .ok_or_else(|| invalid("output row offset overflows host indexing"))?;
            let row = &source[source_start..source_start + packed_row_bytes];
            let output = &mut destination[output_start..output_start + output_row_bytes];
            for (packed_block, decoded_block) in row
                .chunks_exact(block_bytes)
                .zip(output.chunks_exact_mut(block_width * element_bytes))
            {
                let values = &mut block[..block_width];
                decode(packed_block, values);
                write_values(self.metadata.dtype(), values, decoded_block);
            }
        }
        Ok(())
    }

    fn supports_strided_input(&self, input: usize) -> bool {
        input == 1
    }
}

fn write_values(dtype: DataType, values: &[f32], output: &mut [u8]) {
    let bytes = dtype.byte_size();
    for (&value, lane) in values.iter().zip(output.chunks_exact_mut(bytes)) {
        match dtype {
            DataType::Float32 => lane.copy_from_slice(&value.to_ne_bytes()),
            DataType::Float64 => lane.copy_from_slice(&f64::from(value).to_ne_bytes()),
            DataType::Float16 => lane.copy_from_slice(&half::f16::from_f32(value).to_ne_bytes()),
            DataType::BFloat16 => lane.copy_from_slice(&half::bf16::from_f32(value).to_ne_bytes()),
            _ => unreachable!("Metadata validated output dtype"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::dequantize_packed::Format;

    #[test]
    fn normalizes_negative_indices() {
        assert_eq!(normalize_index(-1, 3).unwrap(), 2);
        assert!(normalize_index(-4, 3).is_err());
    }

    #[test]
    fn all_formats_have_a_decoder() {
        let formats = [
            Format::Q4_0,
            Format::Q4_1,
            Format::Q5_0,
            Format::Q5_1,
            Format::Q8_0,
            Format::Q8_1,
            Format::Q2K,
            Format::Q3K,
            Format::Q4K,
            Format::Q5K,
            Format::Q6K,
            Format::Q8K,
            Format::Iq2Xxs,
            Format::Iq2Xs,
            Format::Iq3Xxs,
            Format::Iq1S,
            Format::Iq4Nl,
            Format::Iq3S,
            Format::Iq2S,
            Format::Iq4Xs,
            Format::Iq1M,
            Format::Tq1_0,
            Format::Tq2_0,
            Format::Mxfp4,
            Format::Nvfp4,
            Format::Q1_0,
            Format::Q2_0,
        ];
        for format in formats {
            let _ = decoder(format);
        }
    }
}
