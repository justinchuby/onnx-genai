# Canonical planar auxiliary-scale ABI

- Keep `pkg.nxrt` at version 1 and make the development-stage break atomic.
- `BlockQuantizedMatMul` has four positional inputs: activation, packed weight,
  optional planar scale, and optional bias.
- `BlockQuantizedMoE` has twelve positional inputs: the original slots 0–8 plus
  independent fc1/fc2/fc3 planar scale banks at slots 9–11.
- `block_fp8` uses Float8E4M3FN weights plus Float8E8M0 scales with explicit
  positive 2-D block geometry. `fp4_planar` uses packed Int8 weights plus
  Float8E8M0 scales and fixed `[1,32]` geometry.
- Planar admission is fail-closed and pointer-stable; no compatibility schema,
  dense runtime fallback, or mutable admitted-pointer surface is added.
- Vendored ORT 1.29 rejects Float8E8M0 custom-op inputs during model import,
  before plugin capability discovery. Native CPU/CUDA planar execution remains
  validated; plugin schema/claim/compile/execute is validated with canonical
  interleaved v1 until ORT admits that element type.
- Rebase conflict surface with Sapper:
  `crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_moe.rs`.
