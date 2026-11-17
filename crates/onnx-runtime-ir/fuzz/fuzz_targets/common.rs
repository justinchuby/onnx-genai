//! Shared oracle for the pattern fuzz targets.
//!
//! Any panic, stack overflow, timeout or RSS-limit hit is a defect: the public
//! parse/compile/match entry points must turn every input into `Ok` or a
//! bounded `PatternError` whose span is a valid UTF-8 byte range of the source.

use onnx_runtime_ir::pattern::{
    MatchBudget, MatchOptions, MatchSession, Pattern, PatternError, PatternLibrary,
};
use onnx_runtime_ir::{DataType, Graph, Node, NodeId};

fn operator_exists(domain: &str, op: &str) -> bool {
    domain.is_empty()
        && [
            "Identity",
            "Add",
            "Mul",
            "Relu",
            "Clip",
            "If",
            "Loop",
            "Scan",
            "Split",
            "Dropout",
            "Softmax",
            "Cast",
            "CastLike",
            "QuantizeLinear",
            "DequantizeLinear",
        ]
        .contains(&op)
}

fn check_error(source: &str, error: &PatternError) {
    let span = error.span;
    assert!(
        span.start <= span.end && span.end <= source.len(),
        "span {span:?} is outside source of {} bytes: {error}",
        source.len()
    );
    assert!(
        source.is_char_boundary(span.start) && source.is_char_boundary(span.end),
        "span {span:?} splits a UTF-8 character: {error}"
    );
    assert!(!error.message.is_empty(), "empty diagnostic: {error:?}");
    let _ = error.to_string();
}

/// Small graph covering unary, binary, optional-slot and multi-output nodes.
fn graph() -> Graph {
    let mut graph = Graph::new();
    let a = graph.create_value(DataType::Float32, vec![2.into()]);
    let b = graph.create_value(DataType::Float32, vec![2.into()]);
    graph.set_inputs(vec![a, b]);
    let node = |graph: &mut Graph, op: &str, inputs, outputs: usize| {
        let outs: Vec<_> = (0..outputs)
            .map(|_| graph.create_value(DataType::Float32, vec![2.into()]))
            .collect();
        graph.insert_node(Node::new(NodeId(0), op, inputs, outs.clone()));
        outs
    };
    let s = node(&mut graph, "Add", vec![Some(a), Some(b)], 1)[0];
    let r = node(&mut graph, "Relu", vec![Some(s)], 1)[0];
    let c = node(&mut graph, "Clip", vec![Some(r), None, Some(b)], 1)[0];
    let d = node(&mut graph, "Dropout", vec![Some(c)], 2);
    let m = node(&mut graph, "Mul", vec![Some(d[0]), Some(a)], 1)[0];
    graph.set_outputs(vec![m, d[1]]);
    graph
}

fn exercise(pattern: &Pattern) {
    let graph = graph();
    let session = MatchSession::new(&graph).expect("fixed fuzz graph is valid");
    let options = MatchOptions {
        budget: MatchBudget {
            steps: 20_000,
            candidates: 256,
            tensor_bytes: 1 << 16,
        },
        ..Default::default()
    };
    let _ = session.find_matches(pattern, options);
}

/// Names declared with `pattern NAME` or `interface NAME`, for entry lookup.
fn declared_names(source: &str) -> Vec<&str> {
    let words: Vec<&str> = source
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .collect();
    let mut names: Vec<&str> = words
        .windows(2)
        .filter(|w| w[0] == "pattern" || w[0] == "interface")
        .map(|w| w[1])
        .collect();
    names.sort_unstable();
    names.dedup();
    names.truncate(16);
    names
}

pub fn run(selector: u8, source: &str) {
    let native = selector & 1 == 1;
    if selector & 2 == 0 {
        let parsed = if native {
            Pattern::parse_native(source)
        } else {
            Pattern::parse_ort(source)
        };
        match parsed {
            Ok(pattern) => exercise(&pattern),
            Err(error) => check_error(source, &error),
        }
        return;
    }
    let parsed = if native {
        PatternLibrary::parse_native(source, &operator_exists)
    } else {
        PatternLibrary::parse_ort(source, &operator_exists)
    };
    match parsed {
        Ok(library) => {
            for name in declared_names(source) {
                for entry in [library.pattern(name), library.graph_pattern(name)] {
                    match entry {
                        Ok(pattern) => exercise(&pattern),
                        Err(error) => check_error(source, &error),
                    }
                }
            }
        }
        Err(error) => check_error(source, &error),
    }
}
