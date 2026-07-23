# Native/ORT CUDA parity

Build and run the committed four-model greedy-decode regression:

```bash
source .cudaenv.sh
cargo build --release -p onnx-genai-bench \
  --features bench-native,cuda --bin profile_native
python3 scripts/check_native_ort_parity.py --gpu 0
```

Models default to `~/.foundry/cache/models`; override with `--model-root`.
Select individual cases with repeated `--model`.

Reproduce a Qwen exact-weight float32 oracle with PyTorch, Transformers, ONNX,
and the corresponding HF checkpoint:

```bash
python3 scripts/qwen_q4_f32_oracle.py \
  --case qwen2.5-1.5b \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct
```

The default dequantizes the deployed ONNX Q4 weights to float32. Add
`--weights hf-dense` to compare the original dense checkpoint instead.
