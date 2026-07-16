#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBIUS_ROOT="${MOBIUS_ROOT:-$(dirname "$ROOT")/mobius}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
ARTIFACT_ROOT="${E2E_ARTIFACT_ROOT:-$CARGO_TARGET_DIR/e2e-sub4bit-validation}"
MODEL_DIR="$ARTIFACT_ROOT/model"
MAX_NEW_TOKENS="${E2E_MAX_NEW_TOKENS:-1}"

mkdir -p "$ARTIFACT_ROOT"

if [[ "${E2E_SKIP_EXPORT:-0}" != "1" ]]; then
  PYTHONPATH="$MOBIUS_ROOT/src" python3 "$ROOT/scripts/e2e_sub4bit_export.py" \
    --mobius-root "$MOBIUS_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    | tee "$ARTIFACT_ROOT/export.log"
fi

for required in model.onnx model.onnx.data tokenizer.json inference_metadata.yaml; do
  test -f "$MODEL_DIR/$required" || {
    echo "missing exported artifact: $MODEL_DIR/$required" >&2
    exit 1
  }
done

SHAPE_FILE="$ROOT/crates/onnx-runtime-shape-inference/src/handlers/linalg.rs"
KERNEL_FILE="$ROOT/crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs"
TEST_FILE="$ROOT/crates/onnx-genai-engine/tests/e2e_sub4bit_generated.rs"
BACKUP_DIR="$ARTIFACT_ROOT/source-backups-$$"

if ! git -C "$ROOT" diff --quiet -- "$SHAPE_FILE" "$KERNEL_FILE"; then
  echo "refusing to patch dirty runtime source files" >&2
  exit 1
fi
if [[ -e "$TEST_FILE" ]]; then
  echo "refusing to overwrite $TEST_FILE" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
cp "$SHAPE_FILE" "$BACKUP_DIR/linalg.rs"
cp "$KERNEL_FILE" "$BACKUP_DIR/block_quantized_matmul.rs"

cleanup() {
  cp "$BACKUP_DIR/linalg.rs" "$SHAPE_FILE"
  cp "$BACKUP_DIR/block_quantized_matmul.rs" "$KERNEL_FILE"
  rm -f "$TEST_FILE"
  rm -rf "$BACKUP_DIR"
}
trap cleanup EXIT

# Diagnostic patches are deliberately temporary. They expose two runtime gaps:
# missing shape inference for the registered quantized matmul operators, and no
# built-in execution counter for proving the custom kernel ran.
python3 - "$SHAPE_FILE" "$KERNEL_FILE" <<'PY'
from pathlib import Path
import sys

shape_path = Path(sys.argv[1])
shape = shape_path.read_text()
needle = """/// `com.microsoft.FusedMatMul`: MatMul with optional pre-transposition of the
"""
handler = """/// Shape rule shared by attribute-sized quantized matmul operators.
pub fn attribute_sized_matmul(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(mut shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    let n = ctx
        .node
        .attr("N")
        .and_then(Attribute::as_int)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: ctx.node.op_type.clone(),
            detail: "missing integer N attribute".into(),
        })?;
    let Some(last) = shape.last_mut() else {
        return Err(ShapeInferError::Invalid {
            op: ctx.node.op_type.clone(),
            detail: "A must have rank >= 1".into(),
        });
    };
    *last = DimExpr::constant(n);
    ctx.set_output(0, dtype, shape);
    Ok(())
}

"""
if "pub fn attribute_sized_matmul" not in shape:
    shape = shape.replace(needle, handler + needle, 1)

register = """    reg.register("", "Gemm", 1, gemm);
"""
registrations = """    reg.register("", "Gemm", 1, gemm);
    reg.register(
        "com.github.onnxruntime.genai",
        "BlockQuantizedMatMul",
        1,
        attribute_sized_matmul,
    );
    reg.register("com.microsoft", "MatMulNBits", 1, attribute_sized_matmul);
"""
if '"BlockQuantizedMatMul"' not in shape:
    shape = shape.replace(register, registrations, 1)
shape_path.write_text(shape)

kernel_path = Path(sys.argv[2])
kernel = kernel_path.read_text()
execute = """    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
"""
probe = """    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        static PROBED_IQ4_NL: OnceLock<()> = OnceLock::new();
        static PROBED_IQ4_XS: OnceLock<()> = OnceLock::new();
        let probe = match self.format {
            BlockFormat::Iq4Nl => Some(&PROBED_IQ4_NL),
            BlockFormat::Iq4Xs => Some(&PROBED_IQ4_XS),
            _ => None,
        };
        if std::env::var_os("ONNX_GENAI_BLOCK_QUANT_PROBE").is_some()
            && probe.is_some_and(|probe| probe.set(()).is_ok())
        {
            eprintln!(
                "BLOCK_QUANTIZED_MATMUL_EXECUTED format={:?} K={} N={}",
                self.format, self.k, self.n
            );
        }
"""
if "BLOCK_QUANTIZED_MATMUL_EXECUTED" not in kernel:
    kernel = kernel.replace(execute, probe, 1)
kernel_path.write_text(kernel)
PY

cat > "$TEST_FILE" <<'RS'
use std::{path::PathBuf, time::Instant};

use onnx_genai_engine::{
    GenerateOptions, NativeDecodeDevice, NativeDecodeSession, ProcessorChain,
};
use onnx_genai_ort::Tokenizer;

#[test]
fn real_gguf_sub4bit_generates_coherently() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var("ONNX_GENAI_E2E_SUB4BIT_MODEL")?);
    let prompt = "The capital of France is";
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
    let prompt_tokens = tokenizer.encode(prompt)?;
    let max_new_tokens = std::env::var("ONNX_GENAI_E2E_SUB4BIT_MAX")?.parse()?;
    let options = GenerateOptions {
        max_new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let load_started = Instant::now();
    let mut session =
        NativeDecodeSession::load(model_dir.join("model.onnx"), NativeDecodeDevice::Cpu)?;
    println!("native_load_seconds={:.3}", load_started.elapsed().as_secs_f64());
    println!("kv_layers={}", session.kv_layer_count());

    let generate_started = Instant::now();
    let result =
        session.generate(&prompt_tokens, &options, &ProcessorChain::new(), &tokenizer)?;
    println!("prompt={prompt:?}");
    println!("prompt_tokens={prompt_tokens:?}");
    println!("generated_tokens={:?}", result.token_ids);
    println!("generated_text={:?}", result.text);
    println!(
        "generate_seconds={:.3}",
        generate_started.elapsed().as_secs_f64()
    );
    assert!(
        result.text.to_ascii_lowercase().contains("paris"),
        "incoherent completion: {:?}",
        result.text
    );
    Ok(())
}
RS

echo "Running native CPU generation; temporary diagnostic patches will be restored."
CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
ONNX_GENAI_E2E_SUB4BIT_MODEL="$MODEL_DIR" \
ONNX_GENAI_E2E_SUB4BIT_MAX="$MAX_NEW_TOKENS" \
ONNX_GENAI_BLOCK_QUANT_PROBE=1 \
  cargo test --release \
    --manifest-path "$ROOT/Cargo.toml" \
    -p onnx-genai-engine \
    --features native-backend \
    --test e2e_sub4bit_generated \
    -- --nocapture \
  2>&1 | tee "$ARTIFACT_ROOT/runtime.log"
