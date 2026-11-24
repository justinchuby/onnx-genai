//! CPU reference checks for the corresponding CUDA fixtures; no device is needed.

use half::{bf16, f16};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{DataType, as_static_shape, compute_contiguous_strides};

mod common;
#[path = "fixtures/deform_conv.rs"]
mod deform_conv;
#[path = "fixtures/packed_validation_workspace.rs"]
mod packed_validation_workspace;
#[path = "fixtures/pooling.rs"]
mod pooling;

#[test]
fn cpu_dilated_averagepool_matches_expected_vectors() {
    let ep = CpuExecutionProvider::new();
    for (graph, node_id, input, expected) in pooling::dilated_cases() {
        let node = graph.node(node_id);
        let shape = |id| as_static_shape(&graph.value(id).shape).unwrap();
        let input_shape = shape(node.inputs[0].unwrap());
        let output_shape = shape(node.outputs[0].unwrap());
        let dtype = graph.value(node.inputs[0].unwrap()).dtype;
        let narrow_input: Vec<u16> = match dtype {
            DataType::Float16 => input.iter().map(|&x| f16::from_f32(x).to_bits()).collect(),
            DataType::BFloat16 => input.iter().map(|&x| bf16::from_f32(x).to_bits()).collect(),
            _ => Vec::new(),
        };
        let input_ptr = if dtype == DataType::Float32 {
            input.as_ptr().cast()
        } else {
            narrow_input.as_ptr().cast()
        };
        // All test fixtures use f32 expected values representable in their input dtype.
        let mut got = vec![0.0_f32; expected.len()];
        let mut narrow = vec![0_u16; expected.len()];
        let output_ptr = if dtype == DataType::Float32 {
            got.as_mut_ptr().cast()
        } else {
            narrow.as_mut_ptr().cast()
        };
        let input_strides = compute_contiguous_strides(&input_shape);
        let output_strides = compute_contiguous_strides(&output_shape);
        let input = TensorView::new(
            DevicePtr(input_ptr),
            dtype,
            &input_shape,
            &input_strides,
            ep.device_id(),
        );
        let output = TensorMut::new(
            DevicePtrMut(output_ptr),
            dtype,
            &output_shape,
            &output_strides,
            ep.device_id(),
        );
        ep.get_kernel(
            node,
            std::slice::from_ref(&input_shape),
            graph.opset_imports[""],
        )
        .unwrap()
        .execute(&[input], &mut [output])
        .unwrap();
        if dtype != DataType::Float32 {
            for (value, bits) in got.iter_mut().zip(narrow) {
                *value = if dtype == DataType::Float16 {
                    f16::from_bits(bits).to_f32()
                } else {
                    bf16::from_bits(bits).to_f32()
                };
            }
        }
        pooling::assert_close(&got, &expected, 1e-5);
    }
}

#[test]
fn deform_conv_cpu_matches_fixtures() {
    for c in deform_conv::cases() {
        let out = common::run_cpu(
            "DeformConv",
            "",
            c.opset,
            &c.inputs,
            std::slice::from_ref(&c.output),
            &c.attrs,
        );
        deform_conv::compare(&c, &out[0]);
    }
}

#[test]
fn shared_validation_gpu_fixture_expectations_match_cpu() {
    for (name, domain, opset, inputs, dtype, shape, attrs, expected) in
        packed_validation_workspace::cases()
    {
        let actual =
            common::try_run_cpu(name, domain, opset, &inputs, &[(dtype, shape)], &attrs).unwrap();
        assert_eq!(actual, [expected], "{name}");
    }
}
