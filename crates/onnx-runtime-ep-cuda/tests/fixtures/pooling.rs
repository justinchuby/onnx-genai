use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};

pub fn build_pool_model(
    op_type: &str,
    dtype: DataType,
    input_shape: &[usize],
    output_shape: &[usize],
    kernel_shape: impl AsRef<[i64]>,
    strides: impl AsRef<[i64]>,
    pads: impl AsRef<[i64]>,
    count_include_pad: Option<i64>,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("X", dtype, static_shape(input_shape.iter().copied()));
    graph.add_input(input);
    let output = graph.create_named_value("Y", dtype, static_shape(output_shape.iter().copied()));
    let mut pool = Node::new(NodeId(0), op_type, vec![Some(input)], vec![output]);
    pool.attributes.insert(
        "kernel_shape".into(),
        Attribute::Ints(kernel_shape.as_ref().to_vec()),
    );
    pool.attributes
        .insert("strides".into(), Attribute::Ints(strides.as_ref().to_vec()));
    pool.attributes
        .insert("pads".into(), Attribute::Ints(pads.as_ref().to_vec()));
    if let Some(value) = count_include_pad {
        pool.attributes
            .insert("count_include_pad".into(), Attribute::Int(value));
    }
    let node = graph.insert_node(pool);
    graph.add_output(output);
    (graph, node)
}

pub fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "index {index}: got {got}, expected {expected}"
        );
    }
}

pub fn dilated_cases() -> Vec<(Graph, NodeId, Vec<f32>, Vec<f32>)> {
    let mut cases = Vec::new();
    let mut add = |dtype,
                   input_shape: &[usize],
                   output_shape: &[usize],
                   kernel: &[i64],
                   strides: &[i64],
                   dilations: &[i64],
                   pads: &[i64],
                   auto_pad: Option<&str>,
                   ceil,
                   include_pad,
                   input,
                   expected| {
        let (mut graph, id) = build_pool_model(
            "AveragePool",
            dtype,
            input_shape,
            output_shape,
            kernel,
            strides,
            pads,
            Some(include_pad),
        );
        graph.opset_imports.insert(
            String::new(),
            if dtype == DataType::BFloat16 { 22 } else { 19 },
        );
        let node = graph.node_mut(id);
        node.attributes
            .insert("dilations".into(), Attribute::Ints(dilations.to_vec()));
        node.attributes
            .insert("ceil_mode".into(), Attribute::Int(ceil));
        if let Some(auto_pad) = auto_pad {
            node.attributes.remove("pads");
            node.attributes
                .insert("auto_pad".into(), Attribute::String(auto_pad.into()));
        }
        cases.push((graph, id, input, expected));
    };
    // ONNX v1.16 backend vector test_averagepool_2d_dilations (opset 19).
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        add(
            dtype,
            &[1, 1, 4, 4],
            &[1, 1, 2, 2],
            &[2, 2],
            &[1, 1],
            &[2, 2],
            &[0; 4],
            None,
            1,
            0,
            (1..=16).map(|i| i as f32).collect(),
            vec![6., 7., 10., 11.],
        );
    }
    // Two batches and channels, three spatial axes, mixed dilation, negative values.
    let groups = [0., 20., -20., 10.];
    add(
        DataType::Float32,
        &[2, 2, 4, 3, 5],
        &[2, 2, 2, 2, 2],
        &[2, 2, 2],
        &[1, 1, 1],
        &[2, 1, 3],
        &[],
        Some("VALID"),
        0,
        0,
        groups
            .into_iter()
            .flat_map(|bias| (0..60).map(move |i| i as f32 + bias - 30.))
            .collect(),
        groups
            .into_iter()
            .flat_map(|bias| {
                [-11., -10., -6., -5., 4., 5., 9., 10.]
                    .into_iter()
                    .map(move |x| x + bias)
            })
            .collect(),
    );
    // Explicit asymmetry: a retained partial window and a discarded trailing window.
    for (include_pad, expected) in [(0, vec![2., 3.]), (1, vec![1., 1.5])] {
        add(
            DataType::Float32,
            &[1, 1, 4],
            &[1, 1, 2],
            &[2],
            &[3],
            &[2],
            &[1, 3],
            None,
            1,
            include_pad,
            vec![1., 2., 3., 4.],
            expected,
        );
    }
    // VALID resolves to zero pads but still retains the ceil-mode overhang.
    add(
        DataType::Float32,
        &[1, 1, 4],
        &[1, 1, 2],
        &[2],
        &[2],
        &[2],
        &[],
        Some("VALID"),
        1,
        1,
        vec![1., 2., 7., 10.],
        vec![4., 3.5],
    );
    // Odd total padding splits to different sides for SAME_UPPER and SAME_LOWER.
    for (mode, expected) in [("SAME_UPPER", vec![4., 7.]), ("SAME_LOWER", vec![2., 6.])] {
        add(
            DataType::Float32,
            &[1, 1, 4],
            &[1, 1, 2],
            &[2],
            &[2],
            &[2],
            &[],
            Some(mode),
            0,
            0,
            vec![1., 2., 7., 10.],
            expected,
        );
    }
    cases
}
