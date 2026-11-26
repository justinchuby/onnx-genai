//! Standalone CPU/CUDA MeanVarianceNormalization-29 against analytical goldens.

mod common;

#[path = "fixtures/mean_variance_normalization.rs"]
mod mean_variance_normalization;

use common::require_cuda;
#[cfg(feature = "gpu-tests")]
use mean_variance_normalization::OP;
use mean_variance_normalization::{claims, goldens};
#[cfg(feature = "gpu-tests")]
use onnx_runtime_ep_api::ExecutionProvider;
#[cfg(feature = "gpu-tests")]
use onnx_runtime_ir::{Attribute, DataType};

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn mean_variance_normalization_cuda_goldens_and_parity() {
    let ep = require_cuda();
    claims(&ep);
    goldens(Some(&ep));
}

#[test]
#[ignore = "release CUDA timing probe; requires an isolated GPU"]
fn mean_variance_normalization_cuda_release_probe() {
    #[cfg(feature = "gpu-tests")]
    run_release_probe();
    #[cfg(not(feature = "gpu-tests"))]
    panic!("rebuild with --features gpu-tests before running the CUDA release probe");
}

#[cfg(feature = "gpu-tests")]
fn run_release_probe() {
    use std::time::{Duration, Instant};

    use cudarc::driver::sys::{CUevent_flags, CUgraphExec, cuGraphLaunch};
    use cudarc::driver::{DevicePtr as _, DevicePtrMut as _};
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
    use onnx_runtime_ir::{Node, NodeId, ValueId, compute_contiguous_strides};

    const BATCH: usize = 64;
    const SAMPLES: usize = 5;
    const MAX_DRIFT: f64 = 0.05;
    const MAX_SPREAD: f64 = 0.10;

    let ep = require_cuda();
    let rt = ep.runtime();
    let capture = |shape: &[usize], axes: Vec<i64>| {
        let dtype = DataType::Float32;
        let mut node = Node::new(NodeId(0), OP, vec![Some(ValueId(0))], vec![ValueId(1)]);
        node.attributes.insert("axes".into(), Attribute::Ints(axes));
        let kernel = ep.get_kernel(&node, &[shape.to_vec()], 29).unwrap();
        let n = shape.iter().product::<usize>();
        let strides = compute_contiguous_strides(shape)
            .expect("fixture shape has representable contiguous strides");
        let data: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { -2. } else { 2. }).collect();
        let x = rt.stream().clone_htod(&data).unwrap();
        let mut y = rt.stream().alloc_zeros::<f32>(n).unwrap();
        let (x_ptr, _) = x.device_ptr(rt.stream());
        let (y_ptr, _) = y.device_ptr_mut(rt.stream());
        let inputs = [TensorView::new(
            DevicePtr(x_ptr as *const _),
            dtype,
            shape,
            &strides,
            ep.device_id(),
        )];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(y_ptr as *mut _),
            dtype,
            shape,
            &strides,
            ep.device_id(),
        )];
        kernel.execute(&inputs, &mut outputs).unwrap();
        rt.synchronize().unwrap();

        rt.begin_graph_capture(&[]).unwrap();
        for _ in 0..BATCH {
            kernel.execute(&inputs, &mut outputs).unwrap();
        }
        rt.end_graph_capture().unwrap();
        // Retire the runtime installation but keep the executable and its
        // directly captured tensor buffers alive for the entire sweep.
        let graph = rt.benchmark_graph().expect("retain MVN timing graph");
        assert_eq!(
            graph.node_count().unwrap(),
            BATCH,
            "MVN graph for {shape:?}"
        );
        assert!(rt.reset_graph().expect("detach MVN timing graph"));
        rt.stream().memset_zeros(&mut y).unwrap();
        (graph, x, y, data)
    };

    let anchor = capture(&[512, 1024], vec![-1]);
    let nchw = capture(&[8, 64, 32, 32], vec![0, 2, 3]);
    let global = capture(&[512, 1024], vec![]);
    let stream = rt.stream();
    let start = stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let launch = |exec: CUgraphExec| {
        // SAFETY: all three graph leases and their tensor buffers stay live
        // until all replays on the capture stream are drained.
        unsafe { cuGraphLaunch(exec, rt.stream_ptr()) }
            .result()
            .unwrap();
    };
    let profile = |label: &str, exec| {
        let phase_start = Instant::now();
        let ramp = |duration| {
            let until = Instant::now() + duration;
            while Instant::now() < until {
                launch(exec);
                stream.synchronize().unwrap();
            }
        };
        let samples = || {
            let mut device = [0.; SAMPLES];
            let mut wall = [0.; SAMPLES];
            for i in 0..SAMPLES {
                stream.synchronize().unwrap();
                // CUDA events bracket one replay. Wall time additionally
                // includes host submission and waiting for the end event.
                let wall_start = Instant::now();
                start.record(stream).unwrap();
                launch(exec);
                end.record(stream).unwrap();
                device[i] = f64::from(start.elapsed_ms(&end).unwrap()) * 1_000. / BATCH as f64;
                wall[i] = wall_start.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
                assert!(
                    device[i].is_finite() && device[i] > 0.,
                    "{label}: invalid device sample"
                );
            }
            device.sort_by(f64::total_cmp);
            wall.sort_by(f64::total_cmp);
            (device, wall[SAMPLES / 2])
        };
        let stable = |values: &[f64; SAMPLES]| {
            (values[SAMPLES - 1] - values[0]) / values[SAMPLES / 2] <= MAX_SPREAD
        };
        ramp(Duration::from_secs(8));
        let mut prev = samples().0;
        for _ in 0..8 {
            ramp(Duration::from_millis(250));
            let (device, wall) = samples();
            if (device[SAMPLES / 2] / prev[SAMPLES / 2] - 1.).abs() <= MAX_DRIFT
                && stable(&prev)
                && stable(&device)
            {
                return (device, wall, phase_start.elapsed());
            }
            prev = device;
        }
        panic!("{label}: GPU timing did not converge; rerun on an isolated, stable-clock device");
    };

    let first = profile("row-wise anchor before", anchor.0.raw_exec());
    let per_channel = profile("NCHW per channel", nchw.0.raw_exec());
    let all_axes = profile("global reduction", global.0.raw_exec());
    let last = profile("row-wise anchor after", anchor.0.raw_exec());
    let drift = (last.0[SAMPLES / 2] / first.0[SAMPLES / 2] - 1.).abs();
    assert!(
        drift <= MAX_DRIFT,
        "same-graph anchor drift {:.2}% exceeds {:.0}% (before {:.3} us, after {:.3} us); discard the sweep and rerun on an isolated, stable-clock device",
        drift * 100.,
        MAX_DRIFT * 100.,
        first.0[SAMPLES / 2],
        last.0[SAMPLES / 2],
    );
    for (_, _, y, data) in [anchor, nchw, global] {
        let actual = rt.stream().clone_dtoh(&y).unwrap();
        assert_eq!(actual.len(), data.len());
        assert!(actual.iter().zip(&data).all(|(&y, &x)| y == x / 2.));
    }
    for (label, (us, wall, phase)) in [
        ("row-wise [512, 1024] before", first),
        ("NCHW [8, 64, 32, 32]", per_channel),
        ("global [512, 1024]", all_axes),
        ("row-wise [512, 1024] after", last),
    ] {
        eprintln!(
            "{OP} CUDA {label}: median {:.3} us, range {:.3}-{:.3} us (n={SAMPLES}, {BATCH} nodes/replay); wall {:.3} us/node, phase {:.2} s",
            us[SAMPLES / 2],
            us[0],
            us[SAMPLES - 1],
            wall,
            phase.as_secs_f64(),
        );
    }
    eprintln!("Same-graph anchor drift: {:.2}%", drift * 100.);
}
