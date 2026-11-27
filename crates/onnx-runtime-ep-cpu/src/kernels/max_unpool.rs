use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;
use onnx_runtime_ir::max_unpool::{self as rules, Geometry};

use super::{to_dense_bytes, to_dense_i64, write_dense_bytes};

pub struct MaxUnpoolFactory(pub u64);

impl KernelFactory for MaxUnpoolFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let rank = shapes
            .first()
            .ok_or_else(|| {
                EpError::KernelFailed("MaxUnpool: missing X shape; supply X and I".into())
            })?
            .len();
        let geometry = Geometry::from_node(node, rank).map_err(EpError::KernelFailed)?;
        Ok(Box::new(MaxUnpool {
            version: self.0,
            geometry,
        }))
    }
}

struct MaxUnpool {
    version: u64,
    geometry: Geometry,
}

impl Kernel for MaxUnpool {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let invalid = EpError::KernelFailed;
        if !(2..=3).contains(&inputs.len())
            || inputs[..2].iter().any(TensorView::is_absent)
            || outputs.len() != 1
            || outputs[0].is_absent()
        {
            return Err(invalid(
                "MaxUnpool: supply X, I, optional output_shape and one output".into(),
            ));
        }
        let x = &inputs[0];
        let indices = &inputs[1];
        let width = rules::element_bytes(self.version, x.dtype).map_err(invalid)?;
        rules::validate_i64("I", indices.dtype).map_err(invalid)?;
        rules::validate_indices_shape(x.shape, indices.shape).map_err(invalid)?;
        let explicit = inputs
            .get(2)
            .filter(|input| !input.is_absent())
            .map(|input| {
                rules::validate_i64("output_shape", input.dtype).map_err(invalid)?;
                rules::validate_control_shape(x.shape.len(), input.shape).map_err(invalid)?;
                to_dense_i64(input)
            })
            .transpose()?;
        let expected = self
            .geometry
            .output_shape(x.shape, explicit.as_deref())
            .map_err(invalid)?;
        let output = &mut outputs[0];
        if output.dtype != x.dtype || output.shape != expected.as_slice() {
            return Err(invalid(format!(
                "MaxUnpool: output is {:?} {:?}; bind {:?} {expected:?}",
                output.dtype, output.shape, x.dtype
            )));
        }
        let input_bytes = rules::checked_bytes("X", x.shape, width).map_err(invalid)?;
        rules::checked_bytes("I", indices.shape, 8).map_err(invalid)?;
        let bytes = rules::checked_bytes("output", &expected, width).map_err(invalid)?;
        let elements = bytes / width;
        let mut scattered = Vec::new();
        scattered.try_reserve_exact(bytes).map_err(|error| {
            invalid(format!(
                "MaxUnpool: cannot allocate {bytes} output bytes ({error}); reduce the output shape"
            ))
        })?;
        scattered.resize(bytes, 0);
        if input_bytes != 0 {
            let targets = to_dense_i64(indices)?;
            let data = to_dense_bytes(x)?;
            for (position, (&index, source)) in
                targets.iter().zip(data.chunks_exact(width)).enumerate()
            {
                let target = usize::try_from(index).ok().filter(|&i| i < elements)
                    .ok_or_else(|| invalid(format!("MaxUnpool: I[{position}]={index} is outside [0, {elements}); supply flat output indices")))?;
                let start = target * width;
                scattered[start..start + width].copy_from_slice(source);
            }
        }
        if bytes == 0 {
            return Ok(());
        }
        write_dense_bytes(output, &scattered)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
