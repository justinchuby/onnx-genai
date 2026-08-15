# doug — History

## 2026-07-30T04:10:00Z — ORT-CUDA 1.28 27B baseline

- Delivered the #384 ORT-CUDA 1.28 basic-optimization Qwen3.6-27B INT4 reference: 17.38 tok/s, 57.527 ms/token, and 18,127 MiB peak H200 VRAM.
- Verified both ORT 1.27 and 1.28 abort on the 27B graph under extended/all optimization, isolating the failure to upstream ORT CUDA Level2 optimizer behavior rather than project CUDA kernels.
