//! Structural QDQ recognition and a bounded, single-thread microbenchmark.
//! Run `cargo run --release -p onnx-runtime-ir --example pattern_qdq -- --bench`.
use std::hint::black_box;
use std::time::Instant;

use onnx_runtime_ir::pattern::{MatchOptions, MatchSession, Pattern, PatternError};
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, ValueId, static_shape};

const PATTERN: &str = r"
    (x, packed, scale, zero : tensor[...] | None) => {
        weights = DequantizeLinear(packed, scale, zero)
        return MatMul(x, weights)
    }
";

fn typed_pattern() -> Result<Pattern, PatternError> {
    let mut p = Pattern::builder();
    let [x, packed, scale] = p.inputs(["x", "packed", "scale"]);
    let zero = p.optional_input("zero");
    let weights = p.op("DequantizeLinear", [packed, scale, zero]);
    p.capture("weights", weights);
    let y = p.op("MatMul", [x, weights]);
    p.finish(y)
}

fn graph() -> (Graph, NodeId, NodeId) {
    let mut graph = Graph::new();
    let x = graph.create_value(DataType::Float32, static_shape([1, 16]));
    let packed = graph.create_value(DataType::Uint8, static_shape([16, 16]));
    let scale = graph.create_value(DataType::Float32, vec![]);
    graph.set_inputs(vec![x, packed, scale]);
    let decoded = graph.create_value(DataType::Float32, static_shape([16, 16]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DequantizeLinear",
        vec![Some(packed), Some(scale)],
        vec![decoded],
    ));
    let mut matmul = |weight: ValueId| {
        let output = graph.create_value(DataType::Float32, static_shape([1, 16]));
        graph.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(x), Some(weight)],
            vec![output],
        ))
    };
    let hit = matmul(decoded);
    let miss = matmul(x);
    (graph, hit, miss)
}

fn measure<T>(name: &str, mut work: impl FnMut() -> T) {
    const ITERATIONS: u32 = 2_000;
    for _ in 0..100 {
        black_box(work());
    }
    let mut samples = Vec::new();
    for _ in 0..5 {
        // Synchronous host work including returned-result destruction; excludes
        // graph construction and the 100 untimed warmup iterations.
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(work());
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS));
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{name},{ITERATIONS},5,{:.1},{:.1},{:.1}",
        samples[2], samples[0], samples[4]
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.iter().any(|arg| arg != "--bench") {
        return Err("unsupported argument; run pattern_qdq with no arguments or --bench".into());
    }
    let (graph, hit, miss) = graph();
    let pattern = Pattern::parse_ort(PATTERN)?;
    let typed = typed_pattern()?;
    let session = MatchSession::new(&graph)?;
    let typed_match = session
        .match_at(&typed, hit, MatchOptions::default())?
        .ok_or("typed QDQ pattern did not match; check builder/text structural parity")?;
    let matched = session
        .match_at(&pattern, hit, MatchOptions::default())?
        .ok_or("QDQ fixture did not match; the example's structural contract regressed")?;
    assert_eq!(
        matched.value("packed")?.unwrap().value().dtype,
        DataType::Uint8
    );
    assert_eq!(typed_match.nodes(), matched.nodes());
    for name in ["x", "packed", "scale", "zero", "weights"] {
        assert_eq!(typed_match.value(name)?, matched.value(name)?);
    }
    assert!(
        session
            .match_at(&pattern, miss, MatchOptions::default())?
            .is_none()
    );
    assert!(
        session
            .match_at(&typed, miss, MatchOptions::default())?
            .is_none()
    );
    if arguments.is_empty() {
        println!(
            "Text and typed QDQ structure: {:?}; packed={:?}; zero omitted",
            matched.nodes(),
            matched.value("packed")?.unwrap().id()
        );
        println!(
            "Structural recognition only: validate quantization, shapes and kernel support in an EP predicate."
        );
        return Ok(());
    }
    println!(
        "# arch={} os={} nodes=3 pattern_nodes=2 defaults=true single_thread=true allocations=not_instrumented",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("operation,iterations,samples,median_ns,min_ns,max_ns");
    measure("parse_verify", || {
        Pattern::parse_ort(black_box(PATTERN)).unwrap()
    });
    measure("prepare_graph", || {
        MatchSession::new(black_box(&graph)).unwrap()
    });
    measure("warm_match", || {
        session
            .match_at(black_box(&pattern), hit, MatchOptions::default())
            .unwrap()
    });
    measure("warm_same_op_nonmatch", || {
        session
            .match_at(black_box(&pattern), miss, MatchOptions::default())
            .unwrap()
    });
    let no_root = Pattern::parse_ort("(x)=>return NoSuchOperator(x)")?;
    assert!(
        session
            .find_matches(&no_root, MatchOptions::default())?
            .is_empty()
    );
    measure("indexed_empty_bucket", || {
        session
            .find_matches(black_box(&no_root), MatchOptions::default())
            .unwrap()
    });
    Ok(())
}
