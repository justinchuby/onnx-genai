//! Native CUDA RNN conformance against pinned ONNX expectations and CPU.

mod common;

use common::{
    absent_input, assert_close, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda,
    try_run_cuda,
};
use onnx_runtime_ir::{Attribute, DataType};

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn rnn_pinned_onnx_batchwise_fixture() {
    // onnx/onnx@622adf94d4e720142950aa8178f7357d52e63db0:
    // onnx/backend/test/case/node/rnn.py, test_simple_rnn_batchwise.
    let ep = require_cuda();
    let dtype = DataType::Float32;
    let inputs = [
        float_input(dtype, &[3, 1, 2], &[1., 2., 3., 4., 5., 6.]),
        float_input(dtype, &[1, 4, 2], &[0.5; 8]),
        float_input(dtype, &[1, 4, 4], &[0.5; 16]),
    ];
    let shapes = [(dtype, vec![3, 1, 1, 4]), (dtype, vec![3, 1, 4])];
    let attrs = [
        ("layout", Attribute::Int(1)),
        ("hidden_size", Attribute::Int(4)),
    ];
    let expected = [0.905_148_2, 0.998_177_9, 0.999_966_6]
        .into_iter()
        .flat_map(|x| [x; 4])
        .collect::<Vec<_>>();
    for (backend, outputs) in [
        ("CPU", run_cpu("RNN", "", 19, &inputs, &shapes, &attrs)),
        (
            "CUDA",
            run_cuda(&ep, "RNN", "", 19, &inputs, &shapes, &attrs),
        ),
    ] {
        for output in outputs {
            assert_close(
                backend,
                dtype,
                &decode_floats(&output, dtype),
                &expected,
                1e-6,
            );
        }
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn rnn_bidirectional_lengths_and_clip() {
    let ep = require_cuda();
    for (opset, dtype) in [
        (19, DataType::Float16),
        (21, DataType::Float32),
        (22, DataType::BFloat16),
        (29, DataType::Float64),
    ] {
        let float = |shape: &[usize], values: &[f32]| {
            if dtype == DataType::Float64 {
                let values = values.iter().copied().map(f64::from).collect::<Vec<_>>();
                input(dtype, shape, &values)
            } else {
                float_input(dtype, shape, values)
            }

            #[test]
            #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
            fn rnn_zero_input_width_and_wide_hidden_state() {
                let ep = require_cuda();
                let dtype = DataType::Float32;
                // Wider than one 128-thread block. The last row depends on column zero
                // and the first row depends on the last column, across both timesteps.
                let hidden = 129;
                let mut r = vec![0.; hidden * hidden];
                r[hidden - 1] = 1.;
                r[(hidden - 1) * hidden] = 1.;
                let mut bias = vec![0.; 2 * hidden];
                bias[0] = 1.;
                bias[2 * hidden - 1] = 2.;
                let inputs = [
                    float_input(dtype, &[2, 1, 0], &[]),
                    float_input(dtype, &[1, hidden, 0], &[]),
                    float_input(dtype, &[1, hidden, hidden], &r),
                    float_input(dtype, &[1, 2 * hidden], &bias),
                ];
                let shapes = [(dtype, vec![2, 1, 1, hidden])];
                let attrs = [("activations", Attribute::Strings(vec![b"Relu".to_vec()]))];
                let mut expected = vec![0.; 2 * hidden];
                expected[0] = 1.;
                expected[hidden - 1] = 2.;
                expected[hidden] = 3.;
                expected[2 * hidden - 1] = 3.;
                let output = run_cuda(&ep, "RNN", "", 29, &inputs, &shapes, &attrs);
                assert_eq!(decode_floats(&output[0], dtype), expected);
            }
        };
        let inputs = [
            float(&[2, 3, 1], &[1., 2., 100., 90., 91., 92.]),
            float(&[2, 1, 1], &[1., 1.]),
            float(&[2, 1, 1], &[1., 1.]),
            absent_input(dtype),
            input(DataType::Int32, &[2], &[2_i32, 0]),
            float(&[2, 2, 1], &[0.5, 1., 10., 20.]),
        ];
        let shapes = [(dtype, vec![2, 3, 2, 1]), (dtype, vec![2, 2, 1])];
        let attrs = [
            ("layout", Attribute::Int(1)),
            ("hidden_size", Attribute::Int(1)),
            ("direction", Attribute::String(b"bidirectional".to_vec())),
            (
                "activations",
                Attribute::Strings(vec![b"Relu".to_vec(), b"Affine".to_vec()]),
            ),
            ("activation_alpha", Attribute::Floats(vec![0., 0.5])),
            ("activation_beta", Attribute::Floats(vec![0., 0.25])),
            ("clip", Attribute::Float(3.)),
        ];
        let y = [1.5, 1.625, 3., 1.75, 0., 0., 0., 0., 0., 0., 0., 0.];
        let yh = [3., 1.625, 0., 0.];
        for (backend, outputs) in [
            ("CPU", run_cpu("RNN", "", opset, &inputs, &shapes, &attrs)),
            (
                "CUDA",
                run_cuda(&ep, "RNN", "", opset, &inputs, &shapes, &attrs),
            ),
        ] {
            for (output, expected) in outputs.iter().zip([y.as_slice(), yh.as_slice()]) {
                assert_close(
                    backend,
                    dtype,
                    &decode_floats(output, dtype),
                    expected,
                    1e-6,
                );
            }
        }

        let mut bad = inputs;
        bad[4] = input(DataType::Int32, &[2], &[2_i32, 4]);
        let error = try_run_cuda(&ep, "RNN", "", opset, &bad, &shapes, &attrs).unwrap_err();
        assert!(error.to_string().contains("sequence_lens[1] is 4"));
    }
}
