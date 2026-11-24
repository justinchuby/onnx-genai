use onnx_runtime_loader::{KeyedWeightFile, proto::onnx};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};
use prost::Message;

fn value(name: &str, dims: [i64; 2]) -> onnx::ValueInfoProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};
    onnx::ValueInfoProto {
        name: name.into(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type: 1,
                    shape: Some(onnx::TensorShapeProto {
                        dim: dims
                            .into_iter()
                            .map(|dim| Dimension {
                                value: Some(Value::DimValue(dim)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn check_projection<const K: usize, const N: usize>(device: DevicePreference, cuda: bool) {
    for (tag, kind, expected) in [
        (
            27,
            "F6_E2M3",
            [0.125_f32, -1.5, 7.5, -0.0, -0.25, 2.0, -7.5, 0.875],
        ),
        (
            28,
            "F6_E3M2",
            [0.0625_f32, -1.5, 28.0, -0.0, -0.125, 2.0, -28.0, 0.1875],
        ),
    ] {
        let packed = if tag == 27 {
            [0x01, 0xfb, 0x81, 0x22, 0xf4, 0x1f] // [1,44,31,32,34,16,63,7]
        } else {
            [0x81, 0xfb, 0x81, 0x22, 0xf4, 0x0f] // [1,46,31,32,34,16,63,3]
        };
        let len = K * N * 3 / 4;
        let mut header =
            format!(r#"{{"W":{{"dtype":"{kind}","shape":[{K},{N}],"data_offsets":[0,{len}]}}}}"#);
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        // Vary the pattern across rows, so accidental row aliasing changes the
        // projection rather than reproducing the same tiled weight.
        for row in 0..K {
            let shift = row % 2 * 3;
            bytes.extend(packed.iter().cycle().skip(shift).take(N * 3 / 4));
        }
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        let path = dir.join(format!(
            "float6-projection-{kind}-{cuda}-{K}x{N}-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let model = onnx::ModelProto {
            ir_version: 14,
            opset_import: vec![onnx::OperatorSetIdProto {
                domain: String::new(),
                version: 28,
            }],
            graph: Some(onnx::GraphProto {
                name: "FLOAT6 projection".into(),
                input: vec![value("X", [1, K as i64])],
                output: vec![value("Y", [1, N as i64])],
                initializer: vec![onnx::TensorProto {
                    name: "W".into(),
                    dims: vec![K as i64, N as i64],
                    data_type: tag,
                    data_location: onnx::tensor_proto::DataLocation::External as i32,
                    ..Default::default()
                }],
                node: vec![
                    onnx::NodeProto {
                        op_type: "Cast".into(),
                        input: vec!["W".into()],
                        output: vec!["F".into()],
                        attribute: vec![onnx::AttributeProto {
                            name: "to".into(),
                            i: 1,
                            r#type: onnx::attribute_proto::AttributeType::Int as i32,
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    onnx::NodeProto {
                        op_type: "MatMul".into(),
                        input: vec!["X".into(), "F".into()],
                        output: vec!["Y".into()],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let bench = std::env::var_os("NXRT_TEST_FLOAT6_TIMING").is_some();
        let mut timing_sessions = Vec::new();
        for arm in 0..if bench { 3 } else { 2 } {
            let mut model = model.clone();
            match arm {
                1 => model
                    .graph
                    .as_mut()
                    .unwrap()
                    .output
                    .push(value("F", [K as i64, N as i64])),
                2 => {
                    let graph = model.graph.as_mut().unwrap();
                    graph.node.remove(0);
                    graph.node[0].input[1] = "W".into();
                    let weight = &mut graph.initializer[0];
                    weight.data_type = 1;
                    weight.data_location = 0;
                    weight.raw_data = (0..K)
                        .flat_map(|row| {
                            (0..N).flat_map(move |col| {
                                expected[(col + row % 2 * 4) % 8].to_le_bytes()
                            })
                        })
                        .collect();
                }
                _ => {}
            }
            let builder = InferenceSession::builder()
                .model_bytes(&model.encode_to_vec())
                .device(device.clone());
            #[cfg(feature = "cuda")]
            let timing_ep = (bench && cuda && arm != 1).then(|| {
                use onnx_runtime_ep_api::{EpConfig, ExecutionProvider};
                let mut ep = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0).unwrap();
                ep.initialize(&EpConfig::default()).unwrap();
                std::sync::Arc::new(ep)
            });
            #[cfg(feature = "cuda")]
            let builder = if let Some(ep) = &timing_ep {
                builder.execution_provider(ep.clone())
            } else {
                builder
            };
            let mut session = if arm != 2 {
                builder.keyed_weight_file(KeyedWeightFile::safetensors(&path))
            } else {
                builder
            }
            .build()
            .unwrap();
            assert_eq!(
                session.device_id().device_type == onnx_runtime_ir::DeviceType::Cuda,
                cuda
            );
            assert!(session.execution_provider_fallback_report().is_none());
            assert_eq!(
                session
                    .graph()
                    .nodes
                    .values()
                    .any(|node| node.op_type == "Float6Projection"),
                arm == 0
            );
            for (even, odd) in [(0.5, -0.25), (-0.25, 2.0), (1.0, 0.0)] {
                let x = Tensor::from_f32(&[1, K], &[even, odd].repeat(K / 2)).unwrap();
                let out = session.run(&[("X", &x)]).unwrap()[0].to_vec_f32();
                assert_eq!(out.len(), N);
                for (i, &v) in out.iter().enumerate() {
                    let reference =
                        (even * expected[i % 8] + odd * expected[(i + 4) % 8]) * (K / 2) as f32;
                    assert_eq!(v, reference, "{kind} col {i}");
                }
            }
            let x_values: Vec<_> = (0..K)
                .map(|r| ((r * 173 % 257) as f32 - 128.0) / 151.0)
                .collect();
            let x = Tensor::from_f32(&[1, K], &x_values).unwrap();
            let out = session.run(&[("X", &x)]).unwrap()[0].to_vec_f32();
            assert_eq!(out.len(), N);
            for (col, &actual) in out.iter().enumerate() {
                let terms = x_values
                    .iter()
                    .enumerate()
                    .map(|(row, &x)| f64::from(x) * f64::from(expected[(col + row % 2 * 4) % 8]));
                let reference: f64 = terms.clone().sum();
                let tolerance = 2e-6 * terms.map(f64::abs).sum::<f64>();
                assert!(
                    (f64::from(actual) - reference).abs() <= tolerance,
                    "{kind} col {col}, arm {arm}: got {actual}, expected {reference} ± {tolerance}"
                );
            }
            if cuda && (arm == 0 || (bench && arm == 2)) {
                use onnx_runtime_ir::DataType;
                let mut input = session
                    .allocate_device_binding(
                        "X",
                        None::<String>,
                        DataType::Float32,
                        vec![1, K],
                        vec![1, K],
                    )
                    .unwrap();
                let output = session
                    .allocate_device_output_binding("Y", DataType::Float32, vec![1, N], vec![1, N])
                    .unwrap();
                let bytes: Vec<_> = x_values.iter().flat_map(|x| x.to_le_bytes()).collect();
                input.write_bytes(0, &bytes).unwrap();
                let mut bindings = [input, output];
                assert!(matches!(
                    session
                        .try_capture_with_device_bindings(&[], &mut bindings)
                        .unwrap(),
                    onnx_runtime_session::DeviceGraphCaptureResult::Captured(_)
                ));
                assert_eq!(session.captured_graph_segment_count(), 1);
                let allocations_after_capture =
                    session.device_allocation_counts().unwrap().allocations;
                assert!(session.replay_device_graph(&mut bindings).unwrap());
                let readback = bindings[1].read_bytes().unwrap();
                assert_eq!(readback.len(), out.len() * 4);
                for (bytes, &expected) in readback.chunks_exact(4).zip(&out) {
                    let captured = f32::from_le_bytes(bytes.try_into().unwrap());
                    assert_eq!(captured, expected, "{kind}: captured projection");
                }
                // Deferred releases from earlier eager runs can complete on any replay.
                // The invariant is that replay makes no new allocation requests.
                assert!(session.replay_device_graph(&mut bindings).unwrap());
                assert_eq!(bindings[1].read_bytes().unwrap(), readback);
                assert_eq!(
                    session.device_allocation_counts().unwrap().allocations,
                    allocations_after_capture
                );
                #[cfg(feature = "cuda")]
                if let Some(ep) = &timing_ep {
                    use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
                    let runtime = ep.runtime();
                    let stream = runtime.stream();
                    let context = runtime.cuda_context();
                    let start = context.new_event(Some(CU_EVENT_DEFAULT)).unwrap();
                    let end = context.new_event(Some(CU_EVENT_DEFAULT)).unwrap();
                    const REPLAYS: usize = 256;
                    let mut batch = || {
                        start.record(stream).unwrap();
                        let host = std::time::Instant::now();
                        for _ in 0..REPLAYS {
                            assert!(session.replay_device_graph(&mut bindings).unwrap());
                        }
                        let dispatch_us = host.elapsed().as_secs_f64() * 1e6 / REPLAYS as f64;
                        end.record(stream).unwrap();
                        let stream_us =
                            f64::from(start.elapsed_ms(&end).unwrap()) * 1e3 / REPLAYS as f64;
                        (stream_us, dispatch_us)
                    };
                    let warmup = std::time::Instant::now();
                    while warmup.elapsed() < std::time::Duration::from_secs(8) {
                        std::hint::black_box(batch());
                    }
                    let samples: [_; 5] = std::array::from_fn(|_| batch());
                    eprintln!(
                        "{kind} CUDA K={K} N={N} arm={arm} (stream_us, host_dispatch_us)={samples:.2?}"
                    );
                }
                assert!(session.reset_device_graph().unwrap());
            }
            if bench && arm != 1 {
                timing_sessions.push((session, x));
            }
        }
        if bench {
            assert_eq!(timing_sessions.len(), 2);
            let mut measure = |arm: usize| {
                let (session, x) = &mut timing_sessions[arm];
                // This includes submission and completed transfers, not just kernel time.
                let start = std::time::Instant::now();
                for _ in 0..100 {
                    std::hint::black_box(session.run(&[("X", x)]).unwrap());
                }
                start.elapsed().as_secs_f64() * 1e4
            };
            let warmup = std::time::Instant::now();
            while warmup.elapsed() < std::time::Duration::from_secs(if cuda { 1 } else { 8 }) {
                measure(0);
                measure(1);
            }
            let samples: [_; 6] = std::array::from_fn(|pair| {
                if pair.is_multiple_of(2) {
                    (measure(0), measure(1))
                } else {
                    let expanded = measure(1);
                    (measure(0), expanded)
                }
            });
            eprintln!(
                "{kind} {device:?} K={K} N={N} (packed,expanded) weight_bytes=({len},{}) synchronized_session_us={samples:.1?}",
                K * N * 4
            );
        }
        drop(timing_sessions);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn float6_safetensors_projection_cpu() {
    check_projection::<1024, 2048>(DevicePreference::Cpu, false);
    if std::env::var_os("NXRT_TEST_FLOAT6_TIMING").is_some() {
        check_projection::<4096, 4096>(DevicePreference::Cpu, false);
    } else {
        check_projection::<14, 132>(DevicePreference::Cpu, false);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA GPU and NVRTC")]
fn float6_safetensors_projection_cuda() {
    let device = DevicePreference::Gpu { index: Some(0) };
    check_projection::<1024, 2048>(device.clone(), true);
    if std::env::var_os("NXRT_TEST_FLOAT6_TIMING").is_some() {
        check_projection::<4096, 4096>(device, true);
    } else {
        check_projection::<14, 132>(device, true);
    }
}
