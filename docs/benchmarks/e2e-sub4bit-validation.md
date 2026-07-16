# End-to-end sub-4-bit validation

**Date:** 2026-07-16  
**Result:** Real-model end-to-end generation achieved, with three integration
gaps exposed and worked around by the validation script.

## Model and revisions

- Runtime: `onnx-genai` `bca068c3c406b46adb684d51cbab588ec72a9a2a`
- Exporter: Mobius PR #406,
  `5705eed3722574014be92afca3ee12604a5fbc8d`
- Model: `bartowski/Qwen2.5-0.5B-Instruct-GGUF`,
  `Qwen2.5-0.5B-Instruct-IQ4_XS.gguf` (349,402,688 bytes)
- GGUF SHA-256:
  `df178ccd68e24ce0c74f957765c00e8f5eec5c51216a1acd4444202e58df6cc1`
- Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct`

The GGUF is mixed: 120 `IQ4_NL` projection tensors, 24 `IQ4_XS` down
projections, 24 `Q5_1` value projections, one `Q8_0` embedding, and 121 F32
norm/bias tensors.

## Reproduction

```bash
cd /home/justinchu/onnx-genai-wt-pris
MOBIUS_ROOT=/home/justinchu/mobius-wt-pris \
CARGO_TARGET_DIR=/home/justinchu/onnx-genai/target-pris \
E2E_ARTIFACT_ROOT=/home/justinchu/onnx-genai/target-pris/e2e-sub4bit-full-repro \
E2E_MAX_NEW_TOKENS=16 \
scripts/e2e-sub4bit-validation.sh
```

The script downloads about 350 MB and writes generated artifacts only below
`E2E_ARTIFACT_ROOT`. It exports with Mobius, checks the graph, runs
`NativeDecodeSession` on CPU, injects a one-shot execution probe, and restores
all temporary runtime source changes.

## Exported graph

- 144 `com.github.onnxruntime.genai::BlockQuantizedMatMul` nodes
- Formats: 120 `iq4_nl`, 24 `iq4_xs`
- Every node has `block_layout_version=1` and positive `K`/`N`
- Remaining 24 `Q5_1` value projections are requantized to 4-bit
  `com.microsoft::MatMulNBits`
- Custom opset import: `com.github.onnxruntime.genai` version 1

## Runtime result

Prompt:

```text
The capital of France is
```

Greedy output, 16 tokens:

```text
 Paris. The capital of the United States is Washington, D. C. The
```

Release-mode generation took 49.482 seconds on this host. The execution probe
printed:

```text
BLOCK_QUANTIZED_MATMUL_EXECUTED format=Iq4Nl K=896 N=896
BLOCK_QUANTIZED_MATMUL_EXECUTED format=Iq4Xs K=4864 N=896
```

The native runtime has no operator fallback: successful planning and generation
used the registered CPU kernels. The probe independently confirms both native
formats executed.

## Integration defects found

1. **Mobius mixed-native scaffold selection.** PR #406 chooses the lone `Q8_0`
   embedding as the shared quantization scaffold, then rejects the `Q5_1`
   value projections:

   ```text
   keep_quantized MatMulNBits requantization currently supports only
   4-bit/block-32 targets; got bits=8 block=32
   ```

   The script temporarily selects the intended 4-bit/block-32 scaffold whenever
   native IQ/MXFP4 tensors are present. Native block bytes remain unchanged.

2. **Mobius omits the custom-domain opset import.** The saved model contains
   `BlockQuantizedMatMul` nodes but no matching `opset_import`; the native loader
   correctly rejects it as malformed. The script adds domain version 1 after
   export.

3. **Runtime shape inference is missing for both quantized matmuls.**
   `onnx-runtime-shape-inference` has no rules for
   `BlockQuantizedMatMul` or `MatMulNBits`, so Mobius models with unresolved
   intermediate shapes fail at execution (`Y ... got []`). The script
   temporarily registers the shared rule `A.shape[..-1] + [N]`, then restores
   the source. This is the critical runtime defect blocking an unmodified
   end-to-end run.

The HTTP server currently loads the ORT-backed `Engine::from_dir`; this
validation therefore uses the engine's explicit `native-backend`
`NativeDecodeSession` path, where the Rust CPU operator is registered.
