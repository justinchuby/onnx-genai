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
    let opset_import = vec![OperatorSetIdProto {
        domain: String::new(),
        version: 17,
    }];
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
    for lowering in [WideArgReduceLowering::Direct, WideArgReduceLowering::Tiled] {
        let (model, nodes) = model_from(source_graph(), lowering);
        let name = format!("{lowering:?}").to_lowercase();
        if measure(&name, &model, nodes, &logits).is_none() {
            eprintln!("skipping: no CUDA session available");
            return;
        }
    }
}

/// Edge-case rows the three lowerings must agree on.
///
/// Width is above the tiling threshold so every lowering actually engages, and
/// each case targets one property: tie position, NaN placement, infinities, and
/// a winner in the last element where an off-by-one would hide.
fn parity_cases(vocab: usize) -> Vec<(&'static str, Vec<f32>, bool)> {
    let mut random = Vec::with_capacity(vocab);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..vocab {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        random.push((state >> 40) as f32 / 16_777_216.0 - 0.5);
    }
    let mut spiked = random.clone();
    spiked[vocab / 3] = 9.5;
    vec![
        ("random with a clear maximum", spiked, false),
        ("all equal", vec![2.0; vocab], false),
        (
            "tie across tiles",
            {
                let mut row = vec![0.0; vocab];
                row[5] = 4.0;
                row[vocab - 5] = 4.0;
                row
            },
            false,
        ),
        (
            "maximum in the last element",
            {
                let mut row = vec![-3.0; vocab];
                row[vocab - 1] = -2.0;
                row
            },
            false,
        ),
        (
            "negative infinity everywhere but one",
            {
                let mut row = vec![f32::NEG_INFINITY; vocab];
                row[777] = f32::NEG_INFINITY;
                row[1_234] = -1.0;
                row
            },
            false,
        ),
        (
            "positive infinity",
            {
                let mut row = vec![1.0; vocab];
                row[9_000 % vocab] = f32::INFINITY;
                row
            },
            false,
        ),
        // NaN is undefined for ONNX ArgMax, so the lowerings are allowed to
        // disagree here; the case is kept to record what each one does.
        (
            "leading nan",
            {
                let mut row = vec![f32::NAN; vocab];
                row[4_321] = 1.0;
                row
            },
            true,
        ),
        (
            "nan after the maximum",
            {
                let mut row = vec![0.0; vocab];
                row[100] = 5.0;
                row[200] = f32::NAN;
                row
            },
            true,
        ),
    ]
}

/// Every lowering must select the same token, on whatever runtime is loaded.
///
/// This is the gate for treating the lowerings as interchangeable: it is how a
/// runtime whose arg-reduction was rewritten gets checked against the portable
/// tiling, which depends on nothing but standard ONNX ops.
#[test]
#[ignore = "requires a CUDA device"]
fn every_wide_argmax_lowering_selects_the_same_token() {
    use onnx_genai_engine::pipeline::WideArgReduceLowering;
    let vocab = 16_384usize;
    let lowerings = [WideArgReduceLowering::Direct, WideArgReduceLowering::Tiled];
    println!(
        "runtime: {}",
        onnx_genai_ort::runtime_capability::loaded_version().unwrap_or_else(|| "unknown".into())
    );

    let mut sessions = Vec::new();
    for lowering in lowerings {
        let mut graph = source_graph_with(vocab as i64);
        onnx_genai_engine::pipeline::lower_degenerate_arg_reductions(&mut graph, lowering);
        let model = wrap_model(graph, lowering);
        let name = format!("parity-{lowering:?}").to_lowercase();
        let Some(session) = open_session(&model, &name) else {
            eprintln!("skipping: no CUDA session available");
            return;
        };
        sessions.push((lowering, session));
    }

    let mut disagreements = 0;
    for (name, row, nan_undefined) in parity_cases(vocab) {
        let answers: Vec<(WideArgReduceLowering, i64)> = sessions
            .iter()
            .map(|(lowering, (_environment, session))| (*lowering, run_one(session, vocab, &row)))
            .collect();
        let first = answers[0].1;
        let agreed = answers.iter().all(|(_, token)| *token == first);
        if agreed {
            println!("  {name:38} all -> {first}");
            continue;
        }
        disagreements += 1;
        let detail: Vec<String> = answers
            .iter()
            .map(|(lowering, token)| format!("{lowering:?}={token}"))
            .collect();
        println!("  {name:38} DIFFER {}", detail.join(" "));
        assert!(
            nan_undefined,
            "case {name:?} has a defined answer but the lowerings disagree: {}",
            detail.join(" ")
        );
    }
    println!("cases disagreeing (all NaN-only, which ONNX leaves undefined): {disagreements}");
}

fn source_graph_with(vocab: i64) -> GraphProto {
    let mut graph = source_graph();
    if let Some(type_proto::Value::TensorType(tensor)) = graph.input[0]
        .r#type
        .as_mut()
        .and_then(|kind| kind.value.as_mut())
        && let Some(shape) = tensor.shape.as_mut()
    {
        shape.dim[1].value = Some(tensor_shape_proto::dimension::Value::DimValue(vocab));
    }
    graph
}

fn wrap_model(
    graph: GraphProto,
    lowering: onnx_genai_engine::pipeline::WideArgReduceLowering,
) -> Vec<u8> {
    let _ = lowering;
    let opset_import = vec![OperatorSetIdProto {
        domain: String::new(),
        version: 17,
    }];
    ModelProto {
        ir_version: 8,
        producer_name: "sampler-argmax-parity".into(),
        opset_import,
        graph: Some(graph),
        ..Default::default()
    }
    .encode_to_vec()
}

fn open_session(model: &[u8], name: &str) -> Option<(Environment, Session)> {
    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("argmax-parity");
    std::fs::create_dir_all(&directory).expect("model directory");
    let path = directory.join(format!("{name}.onnx"));
    std::fs::write(&path, model).expect("write model");
    let environment = Environment::new("sampler-argmax-parity").ok()?;
    let options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    let session = Session::new(&environment, &path, options).ok()?;
    Some((environment, session))
}

fn run_one(session: &Session, vocab: usize, row: &[f32]) -> i64 {
    let input = Value::from_slice_f32(row, &[1, vocab as i64]).expect("logits");
    let outputs = session.run(&[("logits", &input)]).expect("run");
    outputs[0].to_vec_i64().expect("token")[0]
}

/// The runtime's own `ArgMax` must match a host reference over many random
/// decode-shaped rows.
///
/// The single-case checks above only prove a lowering is right for the inputs
/// they name. A reduction that splits a row across cooperating blocks can be
/// correct for a row with one clear maximum and wrong for one whose maximum is
/// duplicated, or whose runner-up sits in another block, so this sweeps
/// distributions that make those cases likely at the real decode width.
#[test]
#[ignore = "requires a CUDA device"]
fn direct_argmax_matches_a_host_reference_over_random_rows() {
    use onnx_genai_engine::pipeline::WideArgReduceLowering;
    let vocab = VOCAB as usize;
    let mut graph = source_graph();
    onnx_genai_engine::pipeline::lower_degenerate_arg_reductions(
        &mut graph,
        WideArgReduceLowering::Direct,
    );
    let model = wrap_model(graph, WideArgReduceLowering::Direct);
    let Some((_environment, session)) = open_session(&model, "direct-random") else {
        eprintln!("skipping: no CUDA session available");
        return;
    };
    println!(
        "runtime: {}",
        onnx_genai_ort::runtime_capability::loaded_version().unwrap_or_else(|| "unknown".into())
    );

    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut failures = Vec::new();
    const ROWS: usize = 240;
    for case in 0..ROWS {
        // Quantising to a few distinct levels makes exact ties common, which is
        // where a cross-block reduction has to agree with a serial scan on
        // which index wins.
        let levels = 1u64 << (case % 6 + 1);
        let mut row: Vec<f32> = (0..vocab)
            .map(|_| (next() % levels) as f32 / levels as f32 - 0.5)
            .collect();
        // Half the cases also get a unique maximum somewhere in the row.
        if case % 2 == 0 {
            row[(next() as usize) % vocab] = 8.0;
        }
        let expected = row
            .iter()
            .enumerate()
            .fold(
                (f32::NEG_INFINITY, 0usize),
                |(best, at), (index, &value)| {
                    if value > best {
                        (value, index)
                    } else {
                        (best, at)
                    }
                },
            )
            .1 as i64;
        let got = run_one(&session, vocab, &row);
        if got != expected {
            let ties = row
                .iter()
                .filter(|value| **value == row[expected as usize])
                .count();
            failures.push(format!(
                "case {case}: levels={levels} expected {expected} got {got} \
                 (value at expected {}, value at got {}, {ties} elements share the maximum)",
                row[expected as usize], row[got as usize]
            ));
        }
    }
    if !failures.is_empty() {
        for failure in failures.iter().take(8) {
            println!("  {failure}");
        }
        panic!(
            "{}/{ROWS} random rows selected the wrong index",
            failures.len()
        );
    }
    println!("all {ROWS} random rows matched the host reference");
}
