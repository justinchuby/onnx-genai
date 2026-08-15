use std::error::Error;
use std::path::PathBuf;

use onnx_genai_ort::{DataType, Environment, Session, SessionOptions, Value};

fn main() -> Result<(), Box<dyn Error>> {
    let model = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: qnn_smoke <tiny-static-qdq-model.onnx>")?;
    let env = Environment::new("qnn-smoke")?;
    let session = Session::new(&env, &model, SessionOptions::default())?;
    let input = Value::from_slice_f32(
        &(0..16).map(|value| value as f32).collect::<Vec<_>>(),
        &[1, 1, 4, 4],
    )?;
    let outputs = session.run(&[("input", &input)])?;
    let output = outputs.first().ok_or("model returned no outputs")?;
    if output.dtype() != DataType::Float32 {
        return Err(format!("expected f32 output, got {:?}", output.dtype()).into());
    }
    println!("output={:?}", output.to_vec_f32()?);
    Ok(())
}
