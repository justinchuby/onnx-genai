//! GPU-vs-CPU checks for the CUDA Wave-4 activation kernels.

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain data and the byte slice retains the source lifetime.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_ne_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn run(ep: &CudaExecutionProvider, node: &Node, x: &[f32], bounds: Option<(f32, f32)>) -> Vec<f32> {
    let rt = ep.runtime();
    let dev: DeviceId = ep.device_id();
    let shape = [x.len()];
    let strides = compute_contiguous_strides(&shape);
    let x_buf = ep.allocate(std::mem::size_of_val(x), 256).unwrap();
    let mut y_buf = ep.allocate(std::mem::size_of_val(x), 256).unwrap();
    // SAFETY: x_buf was allocated for exactly this byte slice.
    unsafe { rt.htod(f32_bytes(x), cuptr(x_buf.as_ptr())).unwrap() };

    let x_view = TensorView::new(
        DevicePtr(x_buf.as_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        dev,
    );
    let y_view = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        dev,
    );
    let mut inputs = vec![x_view];
    let mut bound_buffers = Vec::new();
    let scalar_shape: [usize; 0] = [];
    let scalar_strides: [i64; 0] = [];
    if let Some((min, max)) = bounds {
        for value in [min, max] {
            let buf = ep.allocate(4, 256).unwrap();
            // SAFETY: buf is a four-byte allocation for one f32.
            unsafe { rt.htod(f32_bytes(&[value]), cuptr(buf.as_ptr())).unwrap() };
            inputs.push(TensorView::new(
                DevicePtr(buf.as_ptr()),
                DataType::Float32,
                &scalar_shape,
                &scalar_strides,
                dev,
            ));
            bound_buffers.push(buf);
        }
    }

    let kernel = ep.get_kernel(node, &[], 17).unwrap();
    kernel.execute(&inputs, &mut [y_view]).unwrap();
    let mut bytes = vec![0u8; std::mem::size_of_val(x)];
    // SAFETY: y_buf contains x.len() f32 values.
    unsafe { rt.dtoh(&mut bytes, cuptr(y_buf.as_ptr())).unwrap() };

    ep.deallocate(x_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    for buf in bound_buffers {
        ep.deallocate(buf).unwrap();
    }
    bytes_to_f32(&bytes)
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "index {index}: got {got}, want NaN");
        } else {
            assert!(
                (got - want).abs() <= 2e-6,
                "index {index}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn wave4_activations_match_cpu_references() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            return;
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked (library unavailable)");
            return;
        }
    };
    let x = [-3.0, -1.0, -0.0, 0.0, 0.5, 2.0, f32::NAN];
    let node = |op| Node::new(NodeId(0), op, vec![], vec![]);

    assert_close(
        &run(&ep, &node("LeakyRelu"), &x, None),
        &x.map(|v| if v >= 0.0 { v } else { 0.01 * v }),
    );
    assert_close(
        &run(&ep, &node("Elu"), &x, None),
        &x.map(|v| if v >= 0.0 { v } else { v.exp_m1() }),
    );
    assert_close(
        &run(&ep, &node("HardSigmoid"), &x, None),
        &x.map(|v| (0.2 * v + 0.5).clamp(0.0, 1.0)),
    );
    assert_close(
        &run(&ep, &node("Clip"), &x, Some((-1.0, 1.0))),
        &x.map(|v| v.clamp(-1.0, 1.0)),
    );
    assert_close(
        &run(&ep, &node("Softsign"), &x, None),
        &x.map(|v| v / (1.0 + v.abs())),
    );
    assert_close(
        &run(&ep, &node("Selu"), &x, None),
        &x.map(|v| 1.0507 * if v >= 0.0 { v } else { 1.67326 * v.exp_m1() }),
    );
}
