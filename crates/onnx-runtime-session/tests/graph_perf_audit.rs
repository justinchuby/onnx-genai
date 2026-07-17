use std::path::PathBuf;
use std::time::{Duration, Instant};

use onnx_runtime_ir::{
    DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_loader::{
    EpContextDumpConfig, EpContextPartition, Model, dump_ep_context,
};
use onnx_runtime_optimizer::{
    ConstantFolding, DeadNodeElimination, OpFusion, OptimizationPass, PassContext,
};
use onnx_runtime_session::InferenceSession;
use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

const SIZES: [usize; 4] = [1_000, 5_000, 10_000, 20_000];

fn chain_graph(nodes: usize, op_type: &str) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let mut value = graph.create_named_value("input", DataType::Float32, static_shape([1]));
    graph.add_input(value);
    for _ in 0..nodes {
        let output = graph.create_value(DataType::Float32, static_shape([1]));
        graph.insert_node(Node::new(
            NodeId(0),
            op_type,
            vec![Some(value)],
            vec![output],
        ));
        value = output;
    }
    graph.value_mut(value).name = Some("output".to_string());
    graph.add_output(value);
    graph
}

fn silu_chain_graph(nodes: usize) -> Graph {
    assert!(nodes.is_multiple_of(2));
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let mut value = graph.create_named_value("input", DataType::Float32, static_shape([1]));
    graph.add_input(value);
    for _ in 0..nodes / 2 {
        let sigmoid = graph.create_value(DataType::Float32, static_shape([1]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Sigmoid",
            vec![Some(value)],
            vec![sigmoid],
        ));
        let output = graph.create_value(DataType::Float32, static_shape([1]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(value), Some(sigmoid)],
            vec![output],
        ));
        value = output;
    }
    graph.value_mut(value).name = Some("output".to_string());
    graph.add_output(value);
    graph
}

fn wide_graph(nodes: usize, outputs: bool) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([1]));
    graph.add_input(input);
    for i in 0..nodes {
        let output = graph.create_value(DataType::Float32, static_shape([1]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![output],
        ));
        if outputs {
            graph.value_mut(output).name = Some(format!("output_{i}"));
            graph.add_output(output);
        }
    }
    graph
}

fn dce_wide_graph(nodes: usize) -> Graph {
    assert!(nodes > 0);
    let mut graph = wide_graph(nodes - 1, false);
    let input = graph.inputs[0];
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);
    graph
}

fn fusion_pair_graph(nodes: usize) -> Graph {
    assert!(nodes.is_multiple_of(2));
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.create_named_value("a", DataType::Float32, static_shape([1, 1]));
    let w = graph.create_named_value("w", DataType::Float32, static_shape([1, 1]));
    let bias = graph.create_named_value("bias", DataType::Float32, static_shape([1]));
    graph.add_input(a);
    graph.add_input(w);
    graph.add_input(bias);
    for i in 0..nodes / 2 {
        let product = graph.create_value(DataType::Float32, static_shape([1, 1]));
        graph.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(a), Some(w)],
            vec![product],
        ));
        let output =
            graph.create_named_value(format!("output_{i}"), DataType::Float32, static_shape([1, 1]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(product), Some(bias)],
            vec![output],
        ));
        graph.add_output(output);
    }
    graph
}

fn reverse_constant_chain_graph(nodes: usize) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let zero = graph.create_named_value("zero", DataType::Int64, static_shape([1]));
    graph.set_initializer(
        zero,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            0_i64.to_le_bytes().to_vec(),
        )),
    );
    let one = graph.create_named_value("one", DataType::Int64, static_shape([1]));
    graph.set_initializer(
        one,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            1_i64.to_le_bytes().to_vec(),
        )),
    );
    let mut values = Vec::with_capacity(nodes + 1);
    values.push(zero);
    for _ in 0..nodes {
        values.push(graph.create_value(DataType::Int64, static_shape([1])));
    }
    for i in (1..=nodes).rev() {
        graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(values[i - 1]), Some(one)],
            vec![values[i]],
        ));
    }
    graph.value_mut(values[nodes]).name = Some("output".to_string());
    graph.add_output(values[nodes]);
    graph
}

fn repetitions(stage: &str, nodes: usize) -> usize {
    match stage {
        "topological_order" => 7,
        "op_fusion_no_match" => 5,
        _ if nodes <= 1_000 => 3,
        _ if nodes <= 5_000 => 2,
        _ => 1,
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn measure(mut f: impl FnMut() -> Duration, repetitions: usize) -> Duration {
    median((0..repetitions).map(|_| f()).collect())
}

fn print_result(stage: &str, topology: &str, nodes: usize, elapsed: Duration) {
    println!(
        "RESULT,{stage},{topology},{nodes},{:.6}",
        elapsed.as_secs_f64() * 1_000.0
    );
}

#[test]
#[ignore = "manual performance audit"]
fn graph_partition_performance_audit() {
    println!("RESULT,stage,topology,nodes,median_ms");

    for nodes in SIZES {
        let graph = chain_graph(nodes, "Relu");
        let elapsed = measure(
            || {
                let start = Instant::now();
                let order = graph.topological_order().unwrap();
                assert_eq!(order.len(), nodes);
                start.elapsed()
            },
            repetitions("topological_order", nodes),
        );
        print_result("topological_order", "chain", nodes, elapsed);

        let graph = wide_graph(nodes, true);
        let elapsed = measure(
            || {
                let start = Instant::now();
                let order = graph.topological_order().unwrap();
                assert_eq!(order.len(), nodes);
                start.elapsed()
            },
            repetitions("topological_order", nodes),
        );
        print_result("topological_order", "wide_fanout", nodes, elapsed);
    }

    for nodes in SIZES {
        let elapsed = measure(
            || {
                let graph = chain_graph(nodes, "Relu");
                let start = Instant::now();
                let session = InferenceSession::from_graph(graph).unwrap();
                let elapsed = start.elapsed();
                drop(session);
                elapsed
            },
            repetitions("executor_build", nodes),
        );
        print_result("executor_build", "relu_chain", nodes, elapsed);

        let elapsed = measure(
            || {
                let graph = silu_chain_graph(nodes);
                let start = Instant::now();
                let session = InferenceSession::from_graph(graph).unwrap();
                let elapsed = start.elapsed();
                drop(session);
                elapsed
            },
            repetitions("executor_build", nodes),
        );
        print_result("executor_build", "silu_pattern_chain", nodes, elapsed);
    }

    for nodes in SIZES {
        let elapsed = measure(
            || {
                let mut graph = chain_graph(nodes, "Relu");
                let imports = graph.opset_imports.clone();
                let registry = InferenceRegistry::default_registry();
                let start = Instant::now();
                registry
                    .infer_graph(&mut graph, &imports, MergePolicy::Permissive)
                    .unwrap();
                start.elapsed()
            },
            repetitions("shape_inference", nodes),
        );
        print_result("shape_inference", "anonymous_value_chain", nodes, elapsed);
    }

    for nodes in SIZES {
        let elapsed = measure(
            || {
                let mut graph = chain_graph(nodes, "Relu");
                let start = Instant::now();
                OpFusion::new()
                    .run(&mut graph, &PassContext::new())
                    .unwrap();
                assert_eq!(graph.num_nodes(), nodes);
                start.elapsed()
            },
            repetitions("op_fusion_no_match", nodes),
        );
        print_result("op_fusion_no_match", "relu_chain", nodes, elapsed);

        let elapsed = measure(
            || {
                let mut graph = fusion_pair_graph(nodes);
                let start = Instant::now();
                OpFusion::new()
                    .run(&mut graph, &PassContext::new())
                    .unwrap();
                assert_eq!(graph.num_nodes(), nodes / 2);
                start.elapsed()
            },
            repetitions("op_fusion_match_heavy", nodes),
        );
        print_result(
            "op_fusion_match_heavy",
            "independent_matmul_add_pairs",
            nodes,
            elapsed,
        );

        let elapsed = measure(
            || {
                let mut graph = dce_wide_graph(nodes);
                let start = Instant::now();
                DeadNodeElimination
                    .run(&mut graph, &PassContext::new())
                    .unwrap();
                assert_eq!(graph.num_nodes(), 1);
                start.elapsed()
            },
            repetitions("dead_node_elimination", nodes),
        );
        print_result(
            "dead_node_elimination",
            "shared_input_dead_fanout",
            nodes,
            elapsed,
        );

        let elapsed = measure(
            || {
                let mut graph = reverse_constant_chain_graph(nodes);
                let start = Instant::now();
                ConstantFolding
                    .run(&mut graph, &PassContext::new())
                    .unwrap();
                assert_eq!(graph.num_nodes(), 0);
                start.elapsed()
            },
            repetitions("constant_folding", nodes),
        );
        print_result(
            "constant_folding",
            "reverse_node_id_dependency_chain",
            nodes,
            elapsed,
        );
    }

    for nodes in SIZES {
        let elapsed = measure(
            || {
                let graph = wide_graph(nodes, true);
                let covered: Vec<Vec<NodeId>> =
                    graph.nodes.keys().map(|node| vec![node]).collect();
                let partitions: Vec<EpContextPartition<'_>> = covered
                    .iter()
                    .map(|nodes| EpContextPartition {
                        source: "audit",
                        ep_sdk_version: "",
                        partition_name: "",
                        main_context: true,
                        blob: &[],
                        covered_nodes: nodes,
                    })
                    .collect();
                let target_dir = std::env::var_os("CARGO_TARGET_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("target-wallace"));
                std::fs::create_dir_all(&target_dir).unwrap();
                let output = target_dir.join(format!("graph_perf_ctx_{nodes}.onnx"));
                let config = EpContextDumpConfig {
                    enable: true,
                    file_path: Some(output.clone()),
                    embed_mode: 1,
                };
                let model = Model::new(&graph);
                let start = Instant::now();
                dump_ep_context(
                    &model,
                    PathBuf::from("graph_perf_source.onnx").as_path(),
                    &partitions,
                    &config,
                )
                .unwrap();
                let elapsed = start.elapsed();
                std::fs::remove_file(output).unwrap();
                elapsed
            },
            repetitions("epcontext_dump", nodes),
        );
        print_result(
            "epcontext_dump",
            "one_node_partitions_shared_input",
            nodes,
            elapsed,
        );
    }
}
