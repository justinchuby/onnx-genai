//! Shared-lane ownership regression. Compile on Host; execute only with CUDA.
mod common;
#[path = "fixtures/packed_validation_workspace.rs"]
mod fixtures;

use common::raw;
use fixtures::cases;
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, OutputMetadata, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{Attribute, DataType, RUNTIME_DOMAIN, compute_contiguous_strides};

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn affine_moe_publishes_only_validated_output_even_when_down_scales_alias() {
    use DataType::{Float32, Uint8};
    use common::{absent_input, input};

    const TOKENS: usize = 64;
    const OUTPUT_BYTES: usize = TOKENS * 2 * size_of::<f32>();
    const SHIFT: usize = size_of::<f32>();
    const FIRST_SCALES: usize = 3;
    const DOWN_SCALES: usize = 6;

    let ep = common::require_cuda();
    let runtime = ep.runtime();
    let inputs = [
        input(Float32, &[TOKENS, 2], &[1.0f32; TOKENS * 2]),
        input(Float32, &[TOKENS, 1], &[1.0f32; TOKENS]),
        input(Uint8, &[1, 4, 1], &[0x11u8; 4]),
        input(Float32, &[1, 4, 1], &[1.0f32; 4]),
        absent_input(Uint8),
        input(Uint8, &[1, 2, 1], &[0x11u8; 2]),
        input(Float32, &[1, 2, 1], &[1.0f32, 2.0]),
        absent_input(Uint8),
    ];
    let shape = [TOKENS, 2];
    let (graph, node) = common::build_graph(
        "AffineSoftmaxFirstMoE",
        RUNTIME_DOMAIN,
        1,
        &inputs,
        &[(Float32, shape.to_vec())],
        &[
            ("k", Attribute::Int(1)),
            ("first_block", Attribute::Int(2)),
            ("first_zero", Attribute::Int(0)),
            ("down_block", Attribute::Int(2)),
            ("down_zero", Attribute::Int(0)),
        ],
    );
    let shapes: Vec<_> = inputs.iter().map(|input| input.shape.clone()).collect();
    let kernel = ep.get_kernel(graph.node(node), &shapes, 1).unwrap();
    let mut buffers: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let bytes = if index == DOWN_SCALES {
                OUTPUT_BYTES + 2 * SHIFT
            } else {
                input.bytes.len().max(1)
            };
            ep.allocate(bytes, 256).unwrap()
        })
        .collect();
    let pointers: Vec<_> = buffers
        .iter_mut()
        .map(|buffer| buffer.as_mut_ptr())
        .collect();
    let upload = |index: usize, bytes: &[u8]| {
        // The device buffers outlive every copy and dispatch in this test.
        unsafe { runtime.htod(bytes, cuptr(pointers[index])).unwrap() };
    };
    for (index, input) in inputs.iter().enumerate().filter(|(_, input)| !input.absent) {
        upload(index, &input.bytes);
    }
    let strides: Vec<_> = shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape).unwrap())
        .collect();
    let views: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            if input.absent {
                TensorView::absent(input.dtype)
            } else {
                TensorView::new(
                    DevicePtr(pointers[index]),
                    input.dtype,
                    &input.shape,
                    &strides[index],
                    ep.device_id(),
                )
            }
        })
        .collect();
    let mut output = ep.allocate(OUTPUT_BYTES, 256).unwrap();
    let output_ptr = output.as_mut_ptr();
    let output_strides = compute_contiguous_strides(&shape).unwrap();
    let output_at = |ptr, offset| {
        TensorMut::new(
            DevicePtrMut(ptr),
            Float32,
            &shape,
            &output_strides,
            ep.device_id(),
        )
        .with_byte_offset(offset)
    };
    let read = |ptr, len| {
        let mut bytes = vec![0; len];
        unsafe { runtime.dtoh(&mut bytes, cuptr(ptr)).unwrap() };
        bytes
    };
    let mut workspace = None;
    let lane = common::prepare_kernel_workspace(
        &ep,
        kernel.as_ref(),
        &views,
        &[output_at(output_ptr, 0)],
        &mut workspace,
    )
    .unwrap();
    kernel
        .execute_with_workspace(&views, &mut [output_at(output_ptr, 0)], lane)
        .unwrap();
    let expected = read(output_ptr, OUTPUT_BYTES);
    let activated = (2.0f32 / (1.0 + (-2.0f32).exp())) * 2.0;
    for row in common::decode_floats(&expected, Float32).chunks_exact(2) {
        assert!((row[0] - 2.0 * activated).abs() < 1e-5);
        assert!((row[1] - 4.0 * activated).abs() < 1e-5);
    }

    let sentinel = vec![0xa5; OUTPUT_BYTES];
    for slot in [FIRST_SCALES, DOWN_SCALES] {
        upload(slot, &0.0f32.to_ne_bytes());
        unsafe { runtime.htod(&sentinel, cuptr(output_ptr)).unwrap() };
        let err = kernel
            .execute_with_workspace(&views, &mut [output_at(output_ptr, 0)], lane)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("scales in both banks must be positive and finite"),
            "{err}"
        );
        assert_eq!(
            read(output_ptr, OUTPUT_BYTES),
            sentinel,
            "invalid bank at slot {slot} changed output"
        );
        upload(slot, &inputs[slot].bytes);
    }

    // Both output spans overwrite at least one scale used by every token.
    // Distinct per-channel scales and many token warps expose a live read hazard.
    for offset in [0, SHIFT] {
        let mut backing = vec![0xa5; OUTPUT_BYTES + 2 * SHIFT];
        backing[..inputs[DOWN_SCALES].bytes.len()].copy_from_slice(&inputs[DOWN_SCALES].bytes);
        upload(DOWN_SCALES, &backing);
        kernel
            .execute_with_workspace(
                &views,
                &mut [output_at(pointers[DOWN_SCALES], offset)],
                lane,
            )
            .unwrap();
        backing[offset..offset + OUTPUT_BYTES].copy_from_slice(&expected);
        assert_eq!(
            read(pointers[DOWN_SCALES], backing.len()),
            backing,
            "down-scale alias at byte offset {offset}"
        );
    }

    let completed_before = ep.deferred_release_stats().completed;
    let expected_releases = buffers.len() as u64 + 2;
    for buffer in buffers {
        ep.deallocate(buffer)
            .expect("retire affine MoE fixture input backing");
    }
    ep.deallocate(output)
        .expect("retire affine MoE fixture output");
    ep.deallocate_workspace(workspace.expect("affine MoE fixture owns a workspace"))
        .expect("retire affine MoE fixture workspace");
    ep.wait_for_deferred_releases()
        .expect("settle affine MoE fixture releases without quarantined or retained ownership");
    let releases = ep.deferred_release_stats();
    assert_eq!(
        releases.completed,
        completed_before + expected_releases,
        "all affine MoE fixture owners must complete release: {releases:?}"
    );
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
            .map(|shape| {
                compute_contiguous_strides(shape)
                    .expect("fixture shape has representable contiguous strides")
            })
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
        let output_strides = compute_contiguous_strides(&shape)
            .expect("fixture shape has representable contiguous strides");
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
