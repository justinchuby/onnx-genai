use onnx_runtime_ir::{DataType, Graph, Node, NodeId, ValueId, static_shape};
use onnx_runtime_loader::{Model, encode_model};

struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn usize(&mut self, upper: usize) -> usize {
        self.next() as usize % upper
    }
}

fn graph() -> (Graph, ValueId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let hub = graph.create_named_value("hub", DataType::Float32, static_shape([1]));
    graph.add_input(hub);
    for index in 0..32 {
        let output =
            graph.create_named_value(format!("out_{index}"), DataType::Float32, static_shape([1]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(hub), Some(hub)],
            vec![output],
        ));
        graph.add_output(output);
    }
    (graph, hub)
}

fn shuffle(mut values: Vec<(NodeId, u32)>) -> Vec<(NodeId, u32)> {
    let mut state = 0xa54f_f53a_5f1d_36f1u64;
    for index in (1..values.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.swap(index, state as usize % (index + 1));
    }
    values
}

fn reference_remove_node(graph: &mut Graph, node: NodeId) {
    if !graph.nodes.contains(node) {
        return;
    }
    let (input_count, outputs) = {
        let metadata = graph.node(node);
        (metadata.inputs.len(), metadata.outputs.clone())
    };
    for input_index in 0..input_count {
        graph.replace_input(node, input_index, None);
    }
    for &output in &outputs {
        if graph.value(output).producer == Some(node) {
            graph.value_mut(output).producer = None;
        }
    }
    graph.nodes.remove(node);
    for output in outputs {
        let orphan = graph.try_value(output).is_some_and(|value| {
            value.producer.is_none()
                && !graph.has_uses(output)
                && !graph.inputs.contains(&output)
                && !graph.outputs.contains(&output)
                && !graph.initializers.contains_key(&output)
        });
        if orphan {
            graph.values.remove(output);
        }
    }
}

#[test]
fn debug_serialization_and_topology_ignore_consumer_hash_insertion_order() {
    let (first, hub) = graph();
    let (second, _) = graph();
    let mut shuffled = first.clone();
    let uses = shuffled.uses(hub);
    for &(node, input_index) in &uses {
        shuffled.replace_input(node, input_index as usize, None);
    }
    for (node, input_index) in shuffle(uses) {
        shuffled.replace_input(node, input_index as usize, Some(hub));
    }

    let first_bytes = encode_model(&Model::new(&first)).unwrap();
    assert_eq!(first_bytes, encode_model(&Model::new(&second)).unwrap());
    assert_eq!(first_bytes, encode_model(&Model::new(&shuffled)).unwrap());
    assert_eq!(format!("{first:#?}"), format!("{second:#?}"));
    assert_eq!(format!("{first:#?}"), format!("{shuffled:#?}"));
    assert_eq!(first.topological_order(), second.topological_order());
    assert_eq!(first.topological_order(), shuffled.topological_order());
    assert_eq!(first.uses(hub), shuffled.uses(hub));
    assert_eq!(first.consumers(hub), shuffled.consumers(hub));
}

#[test]
fn randomized_single_removal_matches_reference_serialization() {
    let mut rng = TestRng(0x510e_527f_ade6_82d1);
    for trial in 0..250 {
        let mut original = Graph::new();
        original.opset_imports.insert(String::new(), 17);
        let input_count = 1 + rng.usize(3);
        let node_count = 1 + rng.usize(32);
        let mut values = Vec::new();
        for index in 0..input_count {
            let value = original.create_named_value(
                format!("trial_{trial}_input_{index}"),
                DataType::Float32,
                static_shape([1]),
            );
            original.add_input(value);
            values.push(value);
        }
        let mut nodes = Vec::new();
        for index in 0..node_count {
            let inputs = (0..(1 + rng.usize(4)))
                .map(|_| Some(values[rng.usize(values.len())]))
                .collect();
            let output = original.create_named_value(
                format!("trial_{trial}_value_{index}"),
                DataType::Float32,
                static_shape([1]),
            );
            nodes.push(original.insert_node(Node::new(
                NodeId(0),
                "Add",
                inputs,
                vec![output],
            )));
            values.push(output);
        }
        for _ in 0..rng.usize(4) {
            original.add_output(values[rng.usize(values.len())]);
        }
        for index in (1..nodes.len()).rev() {
            nodes.swap(index, rng.usize(index + 1));
        }
        nodes.truncate(rng.usize(nodes.len() + 1));

        let mut actual = original.clone();
        let mut reference = original;
        for node in nodes {
            actual.remove_node(node);
            reference_remove_node(&mut reference, node);
        }

        assert_eq!(
            encode_model(&Model::new(&actual)).unwrap(),
            encode_model(&Model::new(&reference)).unwrap(),
            "serialized model mismatch on trial {trial}"
        );
        assert_eq!(format!("{actual:#?}"), format!("{reference:#?}"));
        assert_eq!(actual.topological_order(), reference.topological_order());
    }
}
