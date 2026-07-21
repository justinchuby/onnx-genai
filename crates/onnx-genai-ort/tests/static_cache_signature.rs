use std::path::{Path, PathBuf};

use onnx_genai_ort::{DataType, Environment, Session, SessionOptions};

fn tiny_scatter_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-scatter/model.onnx.textproto")
}

#[test]
fn tiny_scatter_signature_matches_mobius_static_cache_contract()
-> Result<(), Box<dyn std::error::Error>> {
    let env = Environment::new("tiny-scatter-signature")?;
    let session = Session::new(&env, &tiny_scatter_llm(), SessionOptions::default())?;

    let inputs = session
        .inputs()
        .iter()
        .map(|info| (info.name.as_str(), info.dtype, info.shape.as_slice()))
        .collect::<Vec<_>>();
    assert_eq!(
        inputs,
        [
            ("input_ids", DataType::Int64, &[-1, -1][..]),
            ("key_cache.0", DataType::Float32, &[-1, 16, 16][..]),
            ("value_cache.0", DataType::Float32, &[-1, 16, 16][..]),
            ("write_indices", DataType::Int64, &[-1][..]),
            ("nonpad_kv_seqlen", DataType::Int64, &[-1][..]),
        ]
    );

    let outputs = session
        .outputs()
        .iter()
        .map(|info| (info.name.as_str(), info.dtype, info.shape.as_slice()))
        .collect::<Vec<_>>();
    assert_eq!(
        outputs,
        [
            ("logits", DataType::Float32, &[-1, -1, 32][..]),
            ("updated_key_cache.0", DataType::Float32, &[-1, 16, 16][..]),
            (
                "updated_value_cache.0",
                DataType::Float32,
                &[-1, 16, 16][..]
            ),
        ]
    );

    Ok(())
}
