---
title: CubeCL Execution Providers
aliases:
  - CubeCL EPs
  - cubecl-webgpu
  - cubecl-vulkan
tags:
  - execution
  - ep
  - cubecl
  - webgpu
  - vulkan
  - plugin
status: maintained
lang: zh-CN
created: 2026-08-20
updated: 2026-08-20
---

# CubeCL Execution Providers

> [!summary] 本文回答的问题
> `cubecl-webgpu` 与 `cubecl-vulkan` 如何把同一组 CubeCL kernel 接入 nxrt/ORT,以及它们现在适合承担哪些工作?

CubeCL EP 是一组实验性的 GPU provider,目标是在不直接绑定 CUDA 或厂商 SDK 的
情况下,为原生 `onnx-runtime-*` 执行栈提供可移植的 shader kernel。当前实现覆盖
两个后端:

- `cubecl-webgpu`: 通过 CubeCL 的 WGSL compiler 生成 WGSL,再由 `wgpu` 落到
  Metal、DX12、Vulkan 或浏览器 WebGPU。
- `cubecl-vulkan`: 通过 CubeCL 的 SPIR-V compiler 生成 SPIR-V,并固定走
  Vulkan 路径。

两者共享同一套 EP 契约、算子注册和 kernel 代码;差异被收敛到 `CubeclBackend`
和 CubeCL runtime 类型上,避免运行时悄悄把一个 provider 降级成另一个。

## 为什么需要它

CUDA、CPU 与外部 ORT provider 已经覆盖了许多生产场景,但它们不能回答所有
移植性问题。CubeCL 路径提供了一个中间层:

- kernel 用同一份 Rust/CubeCL 源码表达;
- WebGPU 后端可覆盖 macOS Metal、Windows DX12、Linux Vulkan 等不同图形栈;
- Vulkan 后端保留 SPIR-V/subgroup/dtype 方向的实验空间;
- provider 仍然服从 [[execution/Execution Provider Contract]],由 planner 按
  capability 与 cost 放置节点。

这不是“自动更快”的路径。当前 kernel 面向正确性、可移植性和 EP/ABI 打通,而不
是一个已调优的 GEMM/attention 后端。

## 架构

```text
onnx-genai runtime config
  └─ ONNX_GENAI_EP=cubecl-webgpu | cubecl-vulkan
  └─ ONNX_GENAI_CUBECL_EP_LIB=/path/to/plugin
        ↓
onnx-runtime-ep-cubecl-plugin (cdylib)
  ├─ ORT ABI:  CreateEpFactories / ReleaseEpFactory
  ├─ nxrt ABI: NxrtNegotiate / NxrtCreateEpFactories
  └─ diagnostics: nxrt_ep_* counters
        ↓
onnx-runtime-ep-cubecl
  ├─ CubeclBackend: names, registration names, DeviceType, availability
  ├─ CubeclExecutionProvider<R>
  ├─ CubeclContext<R>: ComputeClient + HandleTable + DeviceId
  └─ kernels: Add / Mul / Relu / MatMul
```

`onnx-runtime-ep-cubecl` 是后端实现 crate。`src/backend.rs` 是名称与可用性的单一
事实来源:

| provider | EP name | ORT registration name | `DeviceType` |
|---|---|---|---|
| `cubecl-webgpu` | `cubecl_webgpu_ep` | `onnxruntime_cubecl_webgpu_ep` | `WebGpu` |
| `cubecl-vulkan` | `cubecl_vulkan_ep` | `onnxruntime_cubecl_vulkan_ep` | `Vulkan` |

`onnx-runtime-ep-cubecl-plugin` 是动态库包装层。它一次导出两个 ABI:ORT plugin EP
使用的 `CreateEpFactories`/`ReleaseEpFactory`,以及 nxrt 原生动态 ABI 使用的
`NxrtNegotiate`/`NxrtCreateEpFactories`。同一个 cdylib 因而既能被 ORT plugin
路径加载,也能被原生 nxrt plugin bridge 加载。

运行时 wiring 还包含两处名称桥接:`onnx-genai-ort` 的 `ep_compat` 把
`cubecl-webgpu`/`cubecl-vulkan` 解析为 `PluginLibrary`,而 `onnx-runtime-ir` 为
Vulkan 增加了独立的 `DeviceType::Vulkan`。nxrt ABI 中 Vulkan 的 device code 是 8,
因此 ABI minor version 从 0 提升到 1。

## 后端与平台矩阵

| provider | Cargo feature | shader compiler | 主要落点 | macOS |
|---|---|---|---|---|
| `cubecl-webgpu` | `webgpu` | WGSL | `wgpu`: Metal、DX12、Vulkan、WebGPU | 支持,通常落到 Metal |
| `cubecl-vulkan` | `vulkan` | SPIR-V | Vulkan | 不可构建 |

`vulkan` feature 会启用 `webgpu` 并打开 `cubecl-wgpu/spirv`。上游
`cubecl-wgpu` 的 `spirv` feature 在 macOS 上是 `cfg(not(target_os = "macos"))`,
因此 `cubecl-vulkan` 在 macOS 上不是“运行时不可用”,而是该 feature 本身不应在
该平台构建。macOS 上需要选择 `cubecl-webgpu`,由 WGSL 路径经 Metal 运行同一组
kernel。

## 构建

这两个 crate 不在 workspace 的 `default-members` 中,因此需要显式指定 package
和 feature。

```bash
cargo build -p onnx-runtime-ep-cubecl --features webgpu
cargo build -p onnx-runtime-ep-cubecl-plugin --features webgpu
```

在非 macOS 的 Vulkan 主机上,构建 SPIR-V/Vulkan 后端:

```bash
cargo build -p onnx-runtime-ep-cubecl --features vulkan
cargo build -p onnx-runtime-ep-cubecl-plugin --features vulkan
```

不带 backend feature 时,crate 仍会编译,但不会构造可用 factory;错误信息会指出
需要 `--features webgpu` 或 `--features vulkan`。

## 选择 provider

运行时配置通过两个变量完成:

```bash
export ONNX_GENAI_CUBECL_EP_LIB=/absolute/path/to/libonnx_runtime_ep_cubecl_plugin.dylib
export ONNX_GENAI_EP=cubecl-webgpu
```

Linux 通常是 `libonnx_runtime_ep_cubecl_plugin.so`,Windows 是
`onnx_runtime_ep_cubecl_plugin.dll`,macOS 是
`libonnx_runtime_ep_cubecl_plugin.dylib`。

也可以选择 Vulkan:

```bash
export ONNX_GENAI_CUBECL_EP_LIB=/absolute/path/to/libonnx_runtime_ep_cubecl_plugin.so
export ONNX_GENAI_EP=cubecl-vulkan
```

`cubecl_webgpu` 与 `cubecl_vulkan` 作为下划线别名被接受。只有当
`ONNX_GENAI_CUBECL_EP_LIB` 指向一个存在的文件时,`selectable_execution_providers()`
才会把 `cubecl-webgpu`/`cubecl-vulkan` 放进可选列表;没有这个路径时,运行时无法
发现仓库内构建出来的 cdylib。

## 当前算子表面

当前 release 只声明非常小的 f32 surface:

| ONNX op | opset 起点 | dtype | 形状限制 |
|---|---:|---|---|
| `Add` | 7 | `Float32` | 相同 shape,或其中一个输入为单元素 scalar |
| `Mul` | 7 | `Float32` | 相同 shape,或其中一个输入为单元素 scalar |
| `Relu` | 6 | `Float32` | contiguous tensor |
| `MatMul` | 9 | `Float32` | `A=[...,M,K]`;`B=[K,N]` 或 batch 与 A 匹配的 `B=[...,K,N]` |

不支持的 dtype、broadcast、rank、batch 或 layout 会被明确拒绝,以便节点落回其他
EP。虽然 CubeCL/Vulkan 方向未来可能支持更多 dtype,当前代码只认领 `Float32`;这样
避免在 feature 或设备能力未探测清楚时提前声明 f16/bf16/int 支持。

## HandleTable 与合成地址

EP API 的 `DeviceBuffer` 和 `TensorView` 把设备内存表示为非空指针,并允许
`TensorView::with_byte_offset` 之类的偏移计算。CUDA 的 `CUdeviceptr` 天然符合这种
模型;CubeCL 的 `Handle` 却是 opaque token,没有可暴露的 device pointer。直接把
`Handle` 的 Rust 地址塞进 `DeviceBuffer` 会在指针加偏移后失效。

`HandleTable` 因此建立了一个虚拟地址表:

- 地址空间从 `1 << 44` 开始,远离常见 host pointer;
- 每个 allocation 以 4096 字节 granule 对齐;
- allocation 之间留一个 guard granule,避免小越界落到下一个 tensor;
- 地址永不复用,free 后仍记录基址,从而把 use-after-free 报成可诊断错误;
- 任一范围内地址都可解析回 `(CubeCL Handle, allocation 内偏移)`。

这个设计让 EP 契约中的指针算术继续成立,但 host 仍然不能解引用这些指针。它们只是
CubeCL buffer 的可检查名字。

## 拷贝与同步限制

Host-to-device 与 device-to-host 通过 CubeCL client 的 `write`/`read_one` 实现。
Device-to-device copy 目前会先读回 host,再写入目标 buffer。原因是 CubeCL 当前没有
暴露 D2D byte-copy API;用一个数值类型 copy kernel 替代也会引入一次完整 dispatch,
并且容易把字节重新解释为错误的元素类型。

这是一项真实成本,尤其会影响包含大量同 EP buffer copy 的图。当前 EP 认为这类 copy
在它所认领的小算子集合中较少见;若后来成为热点,应以专门的 byte/u32 copy kernel 或
上游 D2D API 替换,而不是把成本隐藏在调度模型之外。

## 测试

WebGPU 后端的目标测试命令:

```bash
cargo test -p onnx-runtime-ep-cubecl --features webgpu
```

部分 GPU 测试会在没有合适 adapter 时跳过。设置 `NXRT_REQUIRE_GPU_TESTS=1` 可以把
这些 skip 变成失败,用于带 GPU 的验证环境:

```bash
NXRT_REQUIRE_GPU_TESTS=1 cargo test -p onnx-runtime-ep-cubecl --features webgpu
```

Vulkan 测试应在非 macOS 且具备 Vulkan driver 的主机上使用 `--features vulkan`。

## 正式来源

- [`onnx-runtime-ep-cubecl`](../../crates/onnx-runtime-ep-cubecl/src/lib.rs)
- [`backend.rs`](../../crates/onnx-runtime-ep-cubecl/src/backend.rs)
- [`memory.rs`](../../crates/onnx-runtime-ep-cubecl/src/memory.rs)
- [`provider.rs`](../../crates/onnx-runtime-ep-cubecl/src/provider.rs)
- [`onnx-runtime-ep-cubecl-plugin`](../../crates/onnx-runtime-ep-cubecl-plugin/src/lib.rs)
- [`ep_compat.rs`](../../crates/onnx-genai-ort/src/session/ep_compat.rs)

## 相关笔记

- [[execution/Execution Provider Contract]]
- [[execution/Plugin Execution Providers]]
- [[execution/Execution Backends]]
