use std::any::Any;
use std::sync::Arc;

use half::{bf16, f16};
use onnx_runtime_ep_api::random::Generator;
use onnx_runtime_ep_api::{
    EpError, HostKernelData, HostKernelInput, HostKernelOutput, Kernel, KernelFactory, Result,
    TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

pub struct RandomFactory(pub u64);

impl KernelFactory for RandomFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(RandomKernel(Generator::from_node(node, self.0)?)))
    }
}

struct RandomKernel(Generator);

impl RandomKernel {
    fn generate(&self, dtype: DataType, count: usize) -> Result<Vec<u8>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let size = dtype.byte_size();
        let (key, mut counter) = self.0.reserve(count)?;
        let distribution = self.0.distribution;
        let step = distribution.words_per_sample();
        let mut data = vec![0; count * size];
        for bytes in data.chunks_exact_mut(size) {
            let sample = distribution.sample(key, counter);
            match dtype {
                DataType::Float16 => bytes.copy_from_slice(&f16::from_f64(sample).to_le_bytes()),
                DataType::BFloat16 => bytes.copy_from_slice(&bf16::from_f64(sample).to_le_bytes()),
                DataType::Float32 => bytes.copy_from_slice(&(sample as f32).to_le_bytes()),
                DataType::Float64 => bytes.copy_from_slice(&sample.to_le_bytes()),
                _ => unreachable!("output dtype was validated"),
            }
            counter += step;
        }
        Ok(data)
    }
}

impl Kernel for RandomKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let count = self.0.check_output(inputs, outputs)?;
        if count == 0 {
            return Ok(());
        }
        let output = &mut outputs[0];
        super::write_dense_bytes(output, &self.generate(output.dtype, count)?)
    }

    fn execute_managed_host(
        &self,
        inputs: &[HostKernelInput<'_>],
    ) -> Result<Vec<HostKernelOutput>> {
        let [HostKernelInput::Strings(input)] = inputs else {
            return Err(EpError::KernelFailed(
                "Random*Like: bind exactly one managed string input to supply the output shape"
                    .into(),
            ));
        };
        let (dtype, count) = self.0.host_string_output(input.shape)?;
        Ok(vec![HostKernelOutput {
            shape: input.shape.to_vec(),
            data: HostKernelData::Raw {
                dtype,
                bytes: self.generate(dtype, count)?,
            },
        }])
    }

    fn bind_node_run_state(
        &mut self,
        bind: &mut dyn FnMut(Arc<dyn Any + Send + Sync>) -> Arc<dyn Any + Send + Sync>,
    ) -> Result<()> {
        self.0.bind_node_run_state(bind)
    }
}
