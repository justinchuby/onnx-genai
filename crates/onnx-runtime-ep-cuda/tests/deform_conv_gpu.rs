//! Native DeformConv conformance (no CPU fallback in the GPU harness).

mod common;
#[path = "fixtures/deform_conv.rs"]
mod fixtures;

use common::{assert_close, require_cuda, run_cuda};
use fixtures::{cases, compare};
use onnx_runtime_ir::{Attribute, DataType};

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn deform_conv_cuda_matches_fixtures() {
    let ep = require_cuda();
    for c in cases() {
        let out = run_cuda(
            &ep,
            "DeformConv",
            "",
            c.opset,
            &c.inputs,
            std::slice::from_ref(&c.output),
            &c.attrs,
        );
        compare(&c, &out[0]);
    }
}

#[test]
#[ignore = "release microbenchmark; run under the shared host lock"]
fn deform_conv_release_latency() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use cudarc::driver::sys::CUevent_flags;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
    use onnx_runtime_ep_cpu::CpuExecutionProvider;
    use onnx_runtime_ir::{Node, NodeId, ValueId, compute_contiguous_strides};

    let cpu = CpuExecutionProvider::new();
    let cuda = require_cuda();
    let rt = cuda.runtime();
    for (side, ic, oc, groups, offset_groups) in [
        (24usize, 16, 16, 1, 1),
        (48, 32, 48, 2, 4),
        (64, 64, 64, 8, 4),
        (24, 16, 16, 1, 1),
    ] {
        let spatial = side * side;
        let shapes = vec![
            vec![1, ic, side, side],
            vec![oc, ic / groups, 3, 3],
            vec![1, 18 * offset_groups, side, side],
        ];
        let out_shape = vec![1, oc, side, side];
        let data = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let n = s.iter().product::<usize>();
                (0..n)
                    .map(|p| (((p * 37 + i * 31) % 103) as f32 - 51.0) / 128.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut node = Node::new(
            NodeId(0),
            "DeformConv",
            vec![Some(ValueId(0)), Some(ValueId(1)), Some(ValueId(2))],
            vec![ValueId(3)],
        );
        node.attributes
            .insert("group".into(), Attribute::Int(groups as i64));
        node.attributes
            .insert("offset_group".into(), Attribute::Int(offset_groups as i64));
        node.attributes
            .insert("pads".into(), Attribute::Ints(vec![1; 4]));
        let strides = shapes
            .iter()
            .map(|s| {
                compute_contiguous_strides(s)
                    .expect("fixture shape has representable contiguous strides")
            })
            .collect::<Vec<_>>();
        let out_strides = compute_contiguous_strides(&out_shape)
            .expect("fixture shape has representable contiguous strides");
        let cpu_kernel = cpu.get_kernel(&node, &shapes, 22).unwrap();
        let gpu_kernel = cuda.get_kernel(&node, &shapes, 22).unwrap();
        let dtype = DataType::Float32;
        let host_inputs = data
            .iter()
            .zip(&shapes)
            .zip(&strides)
            .map(|((v, s), st)| {
                TensorView::new(DevicePtr(v.as_ptr().cast()), dtype, s, st, cpu.device_id())
            })
            .collect::<Vec<_>>();
        let gpu_inputs = data
            .iter()
            .map(|v| rt.stream().clone_htod(v).unwrap())
            .collect::<Vec<_>>();
        let mut gpu_out = rt.stream().alloc_zeros::<f32>(oc * spatial).unwrap();
        use cudarc::driver::{DevicePtr as CudaDevicePtr, DevicePtrMut as CudaDevicePtrMut};
        let gpu_views = gpu_inputs
            .iter()
            .zip(&shapes)
            .zip(&strides)
            .map(|((v, s), st)| {
                let (ptr, _guard) = v.device_ptr(rt.stream());
                TensorView::new(DevicePtr(ptr as *const _), dtype, s, st, cuda.device_id())
            })
            .collect::<Vec<_>>();
        let mut host_out = vec![0.0f32; oc * spatial];
        let mut reference = vec![0.0f32; oc * spatial];
        let mut cpu_ms = Vec::new();
        let mut reference_ms = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..3 {
                let mut out = TensorMut::new(
                    DevicePtrMut(host_out.as_mut_ptr().cast()),
                    dtype,
                    &out_shape,
                    &out_strides,
                    cpu.device_id(),
                );
                cpu_kernel
                    .execute(&host_inputs, std::slice::from_mut(&mut out))
                    .unwrap();
                black_box(&host_out);
            }
            cpu_ms.push(start.elapsed().as_secs_f64() * 1e3 / 3.);
            let start = Instant::now();
            direct_reference(side, ic, oc, groups, offset_groups, &data, &mut reference);
            black_box(&reference);
            reference_ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        assert_close(
            "DeformConv CPU benchmark",
            dtype,
            &host_out,
            &reference,
            3e-4,
        );
        let gpu_ms = {
            let (ptr, _guard) = gpu_out.device_ptr_mut(rt.stream());
            let mut out = TensorMut::new(
                DevicePtrMut(ptr as *mut _),
                dtype,
                &out_shape,
                &out_strides,
                cuda.device_id(),
            );
            gpu_kernel
                .execute(&gpu_views, std::slice::from_mut(&mut out))
                .unwrap();
            const BATCH: usize = 32;
            rt.begin_graph_capture(&[]).unwrap();
            for _ in 0..BATCH {
                gpu_kernel
                    .execute(&gpu_views, std::slice::from_mut(&mut out))
                    .unwrap();
            }
            rt.end_graph_capture().unwrap();
            let warmup = Instant::now();
            while warmup.elapsed() < Duration::from_secs(8) {
                rt.replay_graph().unwrap();
                rt.synchronize().unwrap();
            }
            let timings = (0..5)
                .map(|_| {
                    let start = rt
                        .stream()
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    rt.replay_graph().unwrap();
                    let end = rt
                        .stream()
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    f64::from(start.elapsed_ms(&end).unwrap()) / BATCH as f64
                })
                .collect::<Vec<_>>();
            rt.reset_graph().unwrap();
            timings
        };
        let gpu_result = rt.stream().clone_dtoh(&gpu_out).unwrap();
        assert_close(
            "DeformConv CUDA benchmark",
            dtype,
            &gpu_result,
            &reference,
            3e-4,
        );
        let stats = |mut t: Vec<f64>| {
            t.sort_by(f64::total_cmp);
            (t[2], t[0], t[4])
        };
        eprintln!(
            "DeformConv side={side} C={ic} oC={oc} group={groups} offset_group={offset_groups}: median/min/max ms CPU={:?} direct_reference={:?} CUDA_graph={:?}",
            stats(cpu_ms),
            stats(reference_ms),
            stats(gpu_ms)
        );
    }

    fn direct_reference(
        side: usize,
        ic: usize,
        oc: usize,
        groups: usize,
        offset_groups: usize,
        data: &[Vec<f32>],
        out: &mut [f32],
    ) {
        let spatial = side * side;
        let local_c = ic / groups;
        for (index, dst) in out.iter_mut().enumerate() {
            let point = index % spatial;
            let channel = index / spatial;
            let first = (channel / (oc / groups)) * local_c;
            let mut sum = 0.0f64;
            for c in 0..local_c {
                let ic_index = first + c;
                let og = ic_index / (ic / offset_groups);
                for k in 0..9 {
                    let i = (og * 9 + k) * 2 * spatial + point;
                    let v = (point / side) as f64 + (k / 3) as f64 - 1. + f64::from(data[2][i]);
                    let u = (point % side) as f64 + (k % 3) as f64 - 1.
                        + f64::from(data[2][i + spatial]);
                    let mut sampled = 0.;
                    for iy in [v.floor(), v.floor() + 1.] {
                        for ix in [u.floor(), u.floor() + 1.] {
                            if ix >= 0. && iy >= 0. && ix < side as f64 && iy < side as f64 {
                                let j = ic_index * spatial + iy as usize * side + ix as usize;
                                sampled += f64::from(data[0][j])
                                    * (1. - (v - iy).abs())
                                    * (1. - (u - ix).abs());
                            }
                        }
                    }
                    sum += sampled * f64::from(data[1][(channel * local_c + c) * 9 + k]);
                }
            }
            *dst = sum as f32;
        }
    }
}
