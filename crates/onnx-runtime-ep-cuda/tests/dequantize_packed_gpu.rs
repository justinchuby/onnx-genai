mod common;

use common::{input, require_cuda, run_cuda};
use onnx_runtime_ir::DataType;
use onnx_runtime_quantization::dequantize_packed_conformance as oracle;

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn cuda_dequantize_packed_conformance() {
    let ep = require_cuda();
    oracle::check_conformance(|name, packed, shape, dtype| {
        run_cuda(
            &ep,
            "DequantizePacked",
            "",
            29,
            &[input(DataType::Uint8, &[packed.len()], packed)],
            &[(dtype.unwrap_or(DataType::Float32), shape.to_vec())],
            &oracle::attributes(name, shape, dtype),
        )
        .pop()
        .unwrap()
    });
}
