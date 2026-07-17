use std::hint::black_box;
use std::time::{Duration, Instant};

use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};

fn hub_graph(node_count: usize) -> (Graph, Vec<NodeId>) {
    let mut graph = Graph::new();
    let hub = graph.create_value(DataType::Float32, static_shape([1]));
    graph.add_input(hub);
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let output = graph.create_value(DataType::Float32, static_shape([1]));
        nodes.push(graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(hub)],
            vec![output],
        )));
    }
    (graph, nodes)
}

fn sequential_remove(node_count: usize) -> Duration {
    let (mut graph, nodes) = hub_graph(node_count);
    let start = Instant::now();
    for node in nodes {
        graph.remove_node(node);
    }
    let elapsed = start.elapsed();
    black_box(graph);
    elapsed
}

fn single_hub_disconnect(node_count: usize, repeats: usize) -> Duration {
    let (base, nodes) = hub_graph(node_count);
    let target = nodes[0];
    let mut elapsed = Duration::ZERO;
    for _ in 0..repeats {
        let mut graph = base.clone();
        let start = Instant::now();
        graph.remove_node(target);
        elapsed += start.elapsed();
        black_box(graph);
    }
    elapsed / repeats as u32
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("node counts must be positive integers"))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![10_000, 20_000]
    } else {
        sizes
    };

    println!("nodes,sequential_remove_ms,single_hub_disconnect_us");
    for node_count in sizes {
        let remove = median(
            (0..3)
                .map(|_| sequential_remove(node_count))
                .collect::<Vec<_>>(),
        );
        let disconnect = single_hub_disconnect(node_count, 100);
        println!(
            "{node_count},{:.3},{:.3}",
            remove.as_secs_f64() * 1_000.0,
            disconnect.as_secs_f64() * 1_000_000.0
        );
    }
}
