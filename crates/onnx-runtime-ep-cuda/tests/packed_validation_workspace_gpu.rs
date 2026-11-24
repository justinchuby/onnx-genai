//! Shared-lane ownership regression. Compile on Host; execute only with CUDA.
mod common;

use common::{input, raw};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, OutputMetadata, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{Attribute, DataType, RUNTIME_DOMAIN, compute_contiguous_strides};

type Case = (
    &'static str,
    &'static str,
    u64,
    Vec<common::Tensor>,
    DataType,
    Vec<usize>,
    Vec<(&'static str, Attribute)>,
    Vec<u8>,
);

fn cases() -> [Case; 5] {
    use DataType::{Float32, Int32, Int64, Uint8};
    let mut packed_q8 = vec![0u8, 0x3c]; // FP16 scale = 1.
    packed_q8.extend([1u8; 32]);
    [
        (
            "Pack",
            "",
            29,
            vec![input(Uint8, &[4], &[0u8, 1, 2, 3])],
            Uint8,
            vec![1],
            vec![("bits", Attribute::Int(2))],
            vec![0xe4u8],
        ),
        (
            "Unpack",
            "",
            29,
            vec![input(Uint8, &[1], &[0xe4u8]), input(Int64, &[], &[4i64])],
            Uint8,
            vec![4],
            vec![("bits", Attribute::Int(2))],
            vec![0u8, 1, 2, 3],
        ),
        (
            "GroupedMatMul",
            "",
            29,
            vec![
                input(Float32, &[1, 2], &[1f32, 2.]),
                input(Float32, &[1, 2, 1], &[3f32, 4.]),
                input(Int32, &[1, 1], &[0i32]),
            ],
            Float32,
            vec![1, 1, 1],
            vec![],
            raw(&[11f32]),
        ),
        (
            "AffineQuantizedGroupedMatMul",
            RUNTIME_DOMAIN,
            1,
            vec![
                input(Float32, &[1, 2], &[1f32, 2.]),
                input(Uint8, &[1, 1, 1], &[0x43u8]),
                input(Int32, &[1, 1], &[0i32]),
                input(Float32, &[1, 1, 1], &[1f32]),
            ],
            Float32,
            vec![1, 1, 1],
            vec![
                ("depth", Attribute::Int(2)),
                ("block_size", Attribute::Int(2)),
                ("zero", Attribute::Int(0)),
            ],
            raw(&[11f32]),
        ),
        (
            "PackedGroupedMatMul",
            RUNTIME_DOMAIN,
            1,
            vec![
                input(Float32, &[1, 32], &[1f32; 32]),
                input(Uint8, &[34], &packed_q8),
                input(Int32, &[1, 1], &[0i32]),
            ],
            Float32,
            vec![1, 1, 1],
            vec![
                (
                    "format",
                    Attribute::String(
                        onnx_runtime_ir::dequantize_packed::Format::Q8_0
                            .name()
                            .as_bytes()
                            .to_vec(),
                    ),
                ),
                ("shape", Attribute::Ints(vec![1, 1, 32])),
            ],
            raw(&[32f32]),
        ),
    ]
}

#[test]
fn shared_validation_gpu_fixture_expectations_match_cpu() {
    for (name, domain, opset, inputs, dtype, shape, attrs, expected) in cases() {
        let actual =
            common::try_run_cpu(name, domain, opset, &inputs, &[(dtype, shape)], &attrs).unwrap();
        assert_eq!(actual, [expected], "{name}");
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn packed_validation_nodes_share_one_governed_flag_without_private_allocations() {
    let ep = common::require_cuda();
    let runtime = ep.runtime();
    let mut workspace = None;
    let mut lane_origin = None;
    for (name, domain, opset, inputs, dtype, shape, attrs, expected) in cases() {
        let (graph, node) = common::build_graph(
            name,
            domain,
            opset,
            &inputs,
            &[(dtype, shape.clone())],
            &attrs,
        );
        let shapes = inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect::<Vec<_>>();
        let kernel = ep.get_kernel(graph.node(node), &shapes, opset).unwrap();
        let buffers = inputs
            .iter()
            .map(|input| {
                let buffer = ep.allocate(input.bytes.len().max(1), 256).unwrap();
                unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr())).unwrap() };
                buffer
            })
            .collect::<Vec<_>>();
        let strides = shapes
            .iter()
            .map(|shape| compute_contiguous_strides(shape))
            .collect::<Vec<_>>();
        let views = inputs
            .iter()
            .zip(&buffers)
            .zip(&strides)
            .map(|((input, buffer), strides)| {
                TensorView::new(
                    DevicePtr(buffer.as_ptr()),
                    input.dtype,
                    &input.shape,
                    strides,
                    ep.device_id(),
                )
            })
            .collect::<Vec<_>>();
        let mut output = ep.allocate(expected.len().max(1), 256).unwrap();
        let output_strides = compute_contiguous_strides(&shape);
        let mut outputs = [TensorMut::new(
            DevicePtrMut(output.as_mut_ptr()),
            dtype,
            &shape,
            &output_strides,
            ep.device_id(),
        )];
        let requirement = common::runtime_workspace_requirement(
            kernel.as_ref(),
            &views,
            &[OutputMetadata::tensor(dtype, &shape)],
            &outputs,
        )
        .unwrap();
        assert_eq!(requirement.bytes, 4, "{name}");
        let flag = common::prepare_workspace(&ep, requirement, &mut workspace)
            .unwrap()
            .unwrap();
        let origin = flag.ptr().0 as usize;
        assert_eq!(
            *lane_origin.get_or_insert(origin),
            origin,
            "serial nodes must share one owner"
        );
        let before = runtime.allocation_counts();
        let missing = kernel
            .execute_with_workspace(&views, &mut outputs, None)
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains(name) && missing.contains("prepare and pass"),
            "{missing}"
        );
        for _ in 0..3 {
            // Never rely on the previous node's successful status value.
            unsafe { runtime.htod(&[0xff; 4], origin as u64).unwrap() };
            kernel
                .execute_with_workspace(&views, &mut outputs, Some(flag))
                .unwrap();
            let mut actual = vec![0u8; expected.len()];
            unsafe { runtime.dtoh(&mut actual, cuptr(output.as_ptr())).unwrap() };
            assert_eq!(actual, expected, "{name}");
        }
        let (invalid_slot, invalid) = match name {
            "Pack" => (0, vec![4u8, 1, 2, 3]),
            "Unpack" => (1, raw(&[-1i64])),
            _ => (2, raw(&[1i32])), // The fixture has exactly one expert.
        };
        unsafe {
            runtime
                .htod(&invalid, cuptr(buffers[invalid_slot].as_ptr()))
                .unwrap();
        }
        let invalid = kernel
            .execute_with_workspace(&views, &mut outputs, Some(flag))
            .unwrap_err()
            .to_string();
        assert!(
            invalid.contains("outside") || invalid.contains("count"),
            "{name}: {invalid}"
        );
        unsafe {
            runtime
                .htod(
                    &inputs[invalid_slot].bytes,
                    cuptr(buffers[invalid_slot].as_ptr()),
                )
                .unwrap();
        }
        kernel
            .execute_with_workspace(&views, &mut outputs, Some(flag))
            .unwrap();
        let mut recovered = vec![0u8; expected.len()];
        unsafe {
            runtime
                .dtoh(&mut recovered, cuptr(output.as_ptr()))
                .unwrap()
        };
        assert_eq!(
            recovered, expected,
            "{name} must recover on the same borrowed flag"
        );
        assert_eq!(
            runtime.allocation_counts(),
            before,
            "{name} must not allocate/free private status backing"
        );
    }
}
