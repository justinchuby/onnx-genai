use onnx_runtime_ir::{Attribute, DataType};

pub struct Input {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
    pub absent: bool,
}

fn input<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> Input {
    // SAFETY: fixture inputs are plain scalar values with no padding.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    };
    Input {
        dtype,
        shape: shape.to_vec(),
        bytes,
        absent: false,
    }
}

fn absent(dtype: DataType) -> Input {
    Input {
        dtype,
        shape: Vec::new(),
        bytes: Vec::new(),
        absent: true,
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= 1e-4,
            "element {index}: {got} vs {expected}"
        );
    }
}

pub fn assert_cases(
    run: impl Fn(&[Input], &[(DataType, Vec<usize>)], &[(&str, Attribute)]) -> Vec<Vec<u8>>,
) {
    let f = DataType::Float32;
    let tensor = |batch, sequence, values: &[f32]| input(f, &[batch, 1, sequence, 1], values);
    let attrs = |left, right, causal| {
        [
            ("left_window_size", Attribute::Int(left)),
            ("right_window_size", Attribute::Int(right)),
            ("is_causal", Attribute::Int(causal)),
        ]
    };
    let no_cache = [
        tensor(1, 3, &[0.; 3]),
        tensor(1, 4, &[0.; 4]),
        tensor(1, 4, &[2., 4., 8., 16.]),
    ];
    let shape = [(f, vec![1, 1, 3, 1])];
    for (left, right, expected) in [
        (0, 1, [3., 6., 12.]),
        (1, 0, [2., 3., 6.]),
        (-1, 0, [2., 3., 14. / 3.]),
        (0, i64::MAX, [7.5, 28. / 3., 12.]),
    ] {
        let out = run(&no_cache, &shape, &attrs(left, right, 0));
        assert_close(&f32s(&out[0]), &expected);
    }

    let internal = [
        tensor(1, 2, &[1., 2.]),
        tensor(1, 2, &[3., 4.]),
        tensor(1, 2, &[8., 16.]),
        input(DataType::Bool, &[4], &[1u8, 1, 0, 1]),
        tensor(1, 2, &[1., 2.]),
        tensor(1, 2, &[2., 4.]),
    ];
    let output_shapes = [
        (f, vec![1, 1, 2, 1]),
        (f, vec![1, 1, 4, 1]),
        (f, vec![1, 1, 4, 1]),
        (f, vec![1, 1, 2, 4]),
    ];
    for mode in 0..=3 {
        let mut mode_attrs = attrs(1, 2, 1).to_vec();
        mode_attrs.push(("qk_matmul_output_mode", Attribute::Int(mode)));
        let result = run(&internal, &output_shapes, &mode_attrs);
        assert_close(&f32s(&result[0]), &[4., 16.]);
        assert_eq!(f32s(&result[1]), [1., 2., 3., 4.]);
        assert_eq!(f32s(&result[2]), [2., 4., 8., 16.]);
        let negative_infinity = f32::NEG_INFINITY;
        let expected = match mode {
            0 | 1 => [1., 2., 3., 4., 2., 4., 6., 8.],
            2 => [
                negative_infinity,
                2.,
                negative_infinity,
                negative_infinity,
                negative_infinity,
                negative_infinity,
                negative_infinity,
                8.,
            ],
            3 => [0., 1., 0., 0., 0., 0., 0., 1.],
            _ => unreachable!(),
        };
        assert_eq!(f32s(&result[3]), expected);
    }

    let external = [
        tensor(2, 2, &[0.; 4]),
        tensor(2, 5, &[0.; 10]),
        tensor(2, 5, &[2., 4., 8., 16., 256., 32., 64., 128., 256., 512.]),
        input(f, &[5], &[0., f32::NEG_INFINITY, 0., 0., 0.]),
        absent(f),
        absent(f),
        input(DataType::Int64, &[2], &[4i64, 1]),
    ];
    let result = run(&external, &[(f, vec![2, 1, 2, 1])], &attrs(1, 0, 0));
    assert_close(&f32s(&result[0]), &[8., 12., 0., 32.]);
}
