use crate::common::{Tensor, absent_input, assert_close, decode_floats, float_input};
use onnx_runtime_ir::{Attribute, DataType};

pub fn check_opset29_cases(
    run: impl Fn(&[Tensor], &[(DataType, Vec<usize>)], &[(&str, Attribute)]) -> Vec<Vec<u8>>,
) {
    // Frozen upstream ONNX input/output vectors; regeneration lives next to the fixture.
    let mut bytes = include_bytes!("linear_attention29.bin").as_slice();
    let mut count = 0;
    while !bytes.is_empty() {
        let (&[name_len, rule_len, num_outputs], rest) = bytes.split_first_chunk().unwrap();
        let (name, rest) = rest.split_at(name_len.into());
        let label = std::str::from_utf8(name).unwrap();
        let (rule, rest) = rest.split_at(rule_len.into());
        let (&[hq, hk, hv], rest) = rest.split_first_chunk().unwrap();
        bytes = rest;
        let attrs = [
            ("q_num_heads", Attribute::Int(hq.into())),
            ("kv_num_heads", Attribute::Int(hk.into())),
            ("v_num_heads", Attribute::Int(hv.into())),
            ("update_rule", Attribute::String(rule.to_vec())),
        ];
        let mut read_tensor = || {
            let (&rank, rest) = bytes.split_first().unwrap();
            if rank == 0 {
                bytes = rest;
                return absent_input(DataType::Float16);
            }
            let (&width, rest) = rest.split_first().unwrap();
            let dtype = match width {
                2 => DataType::Float16,
                4 => DataType::Float32,
                _ => panic!("invalid fixture element width {width}"),
            };
            let (shape, rest) = rest.split_at(rank.into());
            let shape = shape.iter().map(|&d| usize::from(d)).collect::<Vec<_>>();
            let (data, rest) = rest.split_at(dtype.storage_bytes(shape.iter().product()));
            bytes = rest;
            Tensor {
                dtype,
                shape,
                bytes: data.to_vec(),
                absent: false,
            }
        };
        let inputs = (0..6).map(|_| read_tensor()).collect::<Vec<_>>();
        let expected = (0..num_outputs).map(|_| read_tensor()).collect::<Vec<_>>();
        let outputs = expected
            .iter()
            .map(|t| (t.dtype, t.shape.clone()))
            .collect::<Vec<_>>();
        let got = run(&inputs, &outputs, &attrs);
        assert_eq!(got.len(), expected.len(), "{label}");
        for (i, (actual, expected)) in got.iter().zip(&expected).enumerate() {
            let tol = if expected.dtype == DataType::Float16 {
                2e-3
            } else {
                5e-6
            };
            assert_close(
                &format!("{label} output {i}"),
                expected.dtype,
                &decode_floats(actual, expected.dtype),
                &decode_floats(&expected.bytes, expected.dtype),
                tol,
            );
        }
        if let [_, present, states] = got.as_slice() {
            let [batch, seq, ..] = expected[2].shape.as_slice() else {
                unreachable!()
            };
            if *seq > 0 {
                let per_batch = present.len() / batch;
                for (b, state) in present.chunks_exact(per_batch).enumerate() {
                    let end = (b + 1) * seq * per_batch;
                    assert_eq!(
                        &states[end - per_batch..end],
                        state,
                        "{label}: final snapshot"
                    );
                }
            }
        }
        count += 1;
    }
    assert_eq!(count, 11);

    // V29 also supports legacy GQA, per-key decay, broadcast beta, no past
    // state, and tracks snapshots in the activation dtype when S is omitted.
    let dtype = DataType::BFloat16;
    let inputs = [
        float_input(
            dtype,
            &[1, 2, 8],
            &[
                1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 2., 1., 1., 1., 1., 2.,
            ],
        ),
        float_input(dtype, &[1, 2, 4], &[1., 0., 1., 0., 0., 1., 0., 1.]),
        float_input(dtype, &[1, 2, 2], &[4., 6., 8., 10.]),
        absent_input(dtype),
        float_input(
            dtype,
            &[1, 2, 4],
            &[0., 0., 0., 0., f32::NEG_INFINITY, 0., 0., 0.],
        ),
        float_input(dtype, &[1, 2, 1], &[0.5, 0.5]),
    ];
    let outputs = [
        (dtype, vec![1, 2, 4]),
        (dtype, vec![1, 2, 2, 1]),
        (dtype, vec![1, 2, 2, 2, 1]),
    ];
    let attrs = [
        ("q_num_heads", Attribute::Int(4)),
        ("kv_num_heads", Attribute::Int(2)),
        ("v_num_heads", Attribute::Int(0)),
        ("scale", Attribute::Float(1.0)),
        ("chunk_size", Attribute::Int(1)),
    ];
    let got = run(&inputs, &outputs, &attrs);
    for (bytes, expected) in got.iter().zip([
        &[2., 0., 3., 0., 4., 4., 8., 13.][..],
        &[0., 4., 3., 5.],
        &[2., 0., 3., 0., 0., 4., 3., 5.],
    ]) {
        assert_eq!(decode_floats(bytes, dtype), expected);
    }
}
