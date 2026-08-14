//! Kernel-level comparison of the two degenerate-argmax lowerings.
//!
//! Both models come from the *same* source `ArgMax` graph, lowered by
//! [`tile_degenerate_arg_reductions`] with the fused substitution off and on, so
//! the only difference measured is the lowering. Each is run through a CUDA
//! session as its own island would be, over the decode-sized `[1, 202048]`
//! logits the Muse workflow produces.
//!
//! Run with:
//!
//! ```text
//! cargo test -p onnx-genai-engine --release --features cuda,cuda-13000 \
//!     --test sampler_argmax_bench -- --ignored --nocapture
//! ```
//!
//! For per-kernel attribution rather than totals, run the same binary under
//! `nsys profile -t cuda --cuda-graph-trace=node`.

#![cfg(feature = "cuda")]

use onnx_genai_ort::binding::IoBinding;
use onnx_genai_ort::env::Environment;
use onnx_genai_ort::session::{Session, SessionOptions, ep_selection};
use onnx_genai_ort::value::{DataType, Value};
use onnx_runtime_loader::proto::onnx::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto, TypeProto,
    ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
};
use prost::Message;

const VOCAB: i64 = 202_048;

fn tensor_type(elem: tensor_proto::DataType, dims: &[Option<i64>]) -> TypeProto {
    TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: elem as i32,
            shape: Some(TensorShapeProto {
                dim: dims
                    .iter()
                    .map(|dim| tensor_shape_proto::Dimension {
                        value: Some(match dim {
                            Some(size) => tensor_shape_proto::dimension::Value::DimValue(*size),
                            None => tensor_shape_proto::dimension::Value::DimParam("batch".into()),
                        }),
                        ..Default::default()
                    })
                    .collect(),
            }),
        })),
        ..Default::default()
    }
}

/// The unlowered source: one degenerate last-axis `ArgMax`.
fn source_graph() -> GraphProto {
    GraphProto {
        name: "token_sampler".into(),
        node: vec![NodeProto {
            op_type: "ArgMax".into(),
            name: "argmax".into(),
            input: vec!["logits".into()],
            output: vec!["token".into()],
            attribute: vec![
                onnx_runtime_loader::proto::onnx::AttributeProto {
                    name: "axis".into(),
                    r#type: onnx_runtime_loader::proto::onnx::attribute_proto::AttributeType::Int
                        as i32,
                    i: -1,
                    ..Default::default()
                },
                onnx_runtime_loader::proto::onnx::AttributeProto {
                    name: "keepdims".into(),
                    r#type: onnx_runtime_loader::proto::onnx::attribute_proto::AttributeType::Int
                        as i32,
                    i: 0,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }],
        input: vec![ValueInfoProto {
            name: "logits".into(),
            r#type: Some(tensor_type(
                tensor_proto::DataType::Float,
                &[None, Some(VOCAB)],
            )),
            ..Default::default()
        }],
        output: vec![ValueInfoProto {
            name: "token".into(),
            r#type: Some(tensor_type(tensor_proto::DataType::Int64, &[None])),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn model_from(
    graph: GraphProto,
    lowering: onnx_genai_engine::pipeline::WideArgReduceLowering,
) -> (Vec<u8>, usize) {
    use onnx_genai_engine::pipeline::WideArgReduceLowering;
    let mut graph = graph;
    let rewrites =
        onnx_genai_engine::pipeline::lower_degenerate_arg_reductions(&mut graph, lowering);
    let expected = usize::from(lowering != WideArgReduceLowering::Direct);
    assert_eq!(rewrites.total(), expected, "unexpected lowering outcome");
    let nodes = graph.node.len();
    let mut opset_import = vec![OperatorSetIdProto {
        domain: String::new(),
        version: 17,
    }];
    if rewrites.fused > 0 {
        opset_import.push(OperatorSetIdProto {
            domain: onnx_genai_engine::pipeline::FUSED_ARGMAX_DOMAIN.to_string(),
            version: 1,
        });
    }
    let model = ModelProto {
        ir_version: 8,
        producer_name: "sampler-argmax-bench".into(),
        opset_import,
        graph: Some(graph),
        ..Default::default()
    };
    (model.encode_to_vec(), nodes)
}

fn canonical_logits() -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut values: Vec<f32> = (0..VOCAB as usize)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        })
        .collect();
    values[180_559] = 12.5;
    values
}

/// Time one session over device-resident logits with a stable IO binding, which
/// is how the workflow's sampler island runs it: no per-step host transfer, no
/// per-step rebinding, so what is measured is the lowering rather than the
/// harness.
fn measure(name: &str, model: &[u8], nodes: usize, logits: &[f32]) -> Option<f64> {
    let environment = Environment::new("sampler-argmax-bench").ok()?;
    let options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    let session = Session::from_model_bytes(&environment, name.to_string(), model, options).ok()?;
    let allocator = session.device_allocator().ok()??;

    // Both tensors live on the device for the whole measurement, as they do in
    // the workflow: the decoder leaves its logits there and the sampler leaves
    // its token there. Nothing is transferred per run.
    let device =
        Value::empty_in(&[1, VOCAB], DataType::Float32, &allocator).expect("device logits");
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    onnx_genai_ort::cuda_rt::memcpy_host_to_device(
        device.data_ptr_addr().expect("device pointer"),
        &bytes,
    )
    .expect("upload logits");
    let token = Value::empty_in(&[1], DataType::Int64, &allocator).expect("device token");

    let mut binding = IoBinding::new(&session).expect("io binding");
    binding.bind_input("logits", &device).expect("bind input");
    binding.bind_output("token", &token).expect("bind output");

    for _ in 0..50 {
        session.run_with_binding(&binding).expect("warmup run");
    }
    assert_eq!(
        token
            .to_host_from_cuda(0)
            .expect("download token")
            .to_vec_i64()
            .expect("token"),
        vec![180_559],
        "{name} must select the canonical token"
    );
    const ITERS: u32 = 2_000;
    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        session.run_with_binding(&binding).expect("timed run");
    }
    let each = start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
    println!("{name:>10}: {nodes} node(s), {each:8.2} us per run");
    Some(each)
}

#[test]
#[ignore = "requires a CUDA device"]
fn fused_argmax_beats_the_portable_tiling() {
    use onnx_genai_engine::pipeline::WideArgReduceLowering;
    let logits = canonical_logits();
    let (tiled_model, tiled_nodes) = model_from(source_graph(), WideArgReduceLowering::Tiled);
    let (fused_model, fused_nodes) = model_from(source_graph(), WideArgReduceLowering::Fused);
    assert!(
        fused_nodes < tiled_nodes,
        "the fused lowering must emit fewer nodes"
    );

    let Some(tiled) = measure("tiled", &tiled_model, tiled_nodes, &logits) else {
        eprintln!("skipping: no CUDA session available");
        return;
    };
    let fused = measure("fused", &fused_model, fused_nodes, &logits)
        .expect("the tiled session built, so the fused one must too");
    println!("speedup: {:.2}x", tiled / fused);
    assert!(
        fused < tiled,
        "fused {fused:.2} us must beat tiled {tiled:.2} us"
    );
}

/// Every lowering of the same source graph, on the same device logits.
///
/// This is the comparison that decides whether the workaround is still needed:
/// `direct` is the graph as authored, so on a runtime whose arg-reduction
/// parallelises a wide last axis it should be at least as fast as the
/// lowerings, and on one that does not it is the degenerate case they exist to
/// avoid. Reported, not asserted, because which one wins is a property of the
/// runtime under test.
#[test]
#[ignore = "requires a CUDA device"]
fn report_every_wide_argmax_lowering() {
    use onnx_genai_engine::pipeline::WideArgReduceLowering;
    let logits = canonical_logits();
    println!(
        "runtime: {} | direct-capable: {}",
        onnx_genai_ort::runtime_capability::loaded_version().unwrap_or_else(|| "unknown".into()),
        onnx_genai_ort::runtime_capability::reduces_wide_last_axis_on_cuda()
    );
    for lowering in [
        WideArgReduceLowering::Direct,
        WideArgReduceLowering::Tiled,
        WideArgReduceLowering::Fused,
    ] {
        let (model, nodes) = model_from(source_graph(), lowering);
        let name = format!("{lowering:?}").to_lowercase();
        if measure(&name, &model, nodes, &logits).is_none() {
            eprintln!("skipping: no CUDA session available");
            return;
        }
    }
}
