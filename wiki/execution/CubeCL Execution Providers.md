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

当前 release 声明一个很小的 surface,f32 与 f16 两种 dtype:

| ONNX op | opset 起点 | dtype | 形状限制 |
|---|---:|---|---|
| `Add` | 7 | `Float32`, `Float16`\* | 相同 shape,或其中一个输入为单元素 scalar |
| `Mul` | 7 | `Float32`, `Float16`\* | 相同 shape,或其中一个输入为单元素 scalar |
| `Relu` | 6 | `Float32`, `Float16`\* | contiguous tensor |
| `MatMul` | 9 | `Float32`, `Float16`\* | `A=[...,M,K]`;`B=[K,N]` 或 batch 与 A 匹配的 `B=[...,K,N]` |

\* f16 取决于设备探测,见下节。

不支持的 dtype、broadcast、rank、batch 或 layout 会被明确拒绝,以便节点落回其他
EP。一个节点内的所有 float 操作数必须是同一 dtype:kernel 是单态编译的,混合 dtype
会把一个操作数的字节按另一个类型重新解释。

### f16 取决于运行时探测,不是编译期常量

`shader-f16` 在 WebGPU baseline 里是**可选** feature。因此 f16 的可用性由
`runtime::supports_f16()` 在打开设备时探测,而不是假设:

```rust
client.properties().type_usage(ElemType::Float(FloatKind::F16).into())
```

必须同时包含 `TypeUsage::Buffer` 和 `TypeUsage::Arithmetic`。用
`supports_type()` 是不够的 —— 它的语义是「以任何方式被支持」,而 cubecl-wgpu 的
Vulkan 后端就存在只注册 `Conversion` 而不注册 `Buffer` 的 dtype(bf16)。一个只能转换
不能做 buffer 的类型,对我们毫无用处。

探测结果会一路传到宿主:plugin 的 `kernel_registry_entries(f16_available)` 按探测
结果选择要 advertise 的 dtype 集合。**同一个 plugin 二进制在支持和不支持 f16 的适配器
上会声明不同的算子表面**,这样宿主不会先被告知支持 f16、再被节点拒绝打脸。

`ONNX_GENAI_EP=cubecl-webgpu` 下,设备缺 `shader-f16` 与 EP 未实现某 dtype 是两条
不同的拒绝消息,便于区分是硬件限制还是功能缺口。

### MatMul 用 f32 累加

f16 MatMul 的 staging tile 保持 f16(带宽收益的来源),但累加器是 f32。原因是 f16 的
running total 超过 2048 之后 ulp 就变成 2,再加 1.0 会 round-to-even 直接掉回去,长
K 会永久卡住。

这一点有专门的判别性测试 `f16_matmul_accumulates_in_f32`(K=4096,输入全 1.0):f32
累加精确得到 4096,f16 累加实测卡在 2048。**曾经的一个 K=256 版本经负面对照验证过没有
判别力** —— 把累加器改回 f16 它照样通过。改测试之前请先确认新版本能在 kernel 退化时
失败。

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

## 已知成本:形状特化会触发 shader 编译

MatMul kernel 的 `m`/`n`/`k`/`rhs_batched` 是 `#[comptime]` 参数,它们会进入
CubeCL 的 `KernelId::info`。所以:

- **同一形状重复调用命中 kernel 缓存**,只编译一次;
- **每出现一个新形状就编译一个新 shader**。

对固定形状的推理这是一次性成本。对 seq 长度变化的 LLM decode 就不是 —— 每个新的
`M` 都是一次新编译。这是当前实现的一个真实限制,尚未测量其代价。要消除它需要把
`m`/`n`/`k` 改成运行时参数(放弃 comptime 折叠带来的边界检查消除),或者对形状分桶。
在有实测数据之前不要假设哪一边更划算。

## 生命周期:provider 从构造起就可用

**ORT 的 plugin EP ABI 没有 `initialize` 钩子。** ORT 拿到 factory 之后直接
`get_kernel`。所以 provider 的所有设备资源 —— CubeCL client、device handle、f16
探测结果 —— 全部在构造函数里就绪,`initialize()` 是一个接受但什么都不做的空操作
(nxrt 路径会调它,调了也不该被惩罚)。

这里踩过一次坑,值得记下来:早期版本有一个 `require_initialized` 门,要求先调
`initialize()` 才能 dispatch。它保护的东西其实并不存在(资源本来就在构造时就绪),
但它让**真实 ORT 下每一个节点都失败**:

```text
Compile: get_kernel failed for node '' (Add): kernel execution failed:
cubecl_webgpu_ep: get_kernel was called before initialize(); ...
```

而当时 crate 内的 GPU 测试全绿 —— 因为测试 harness 自己调了 `initialize()`,走了一条
ORT 永远不会走的路径。

现在的门是 `require_live`,语义反过来:provider 构造即 live,只有 `shutdown()` 之后
才拒绝调用(那时 client 已经 drain,继续 dispatch 会撞上被回收的 buffer)。
`shutdown` 之后**不支持**重新 `initialize` 拉起 —— CubeCL client 不可原地重启。

测试 harness 现在**故意不调** `initialize()`,并有一个
`a_freshly_constructed_provider_dispatches_without_initialize` 直接锁住这个契约。

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

### crate 内测试覆盖不到的东西

crate 内的 GPU 测试**不经过 ORT**,它们直接调用 provider。这一层曾经漏掉一个让真实
ORT 下每个节点都失败的 bug(见「生命周期」一节)。所以改动 provider 的生命周期、
dtype advertise 或 plugin ABI 表面时,crate 内测试全绿**不构成**端到端证据,需要另外
用真实 ORT 加载 cdylib 验证:

```bash
cargo build --release -p onnx-runtime-ep-cubecl-plugin --features webgpu
python scripts/bench_cubecl_vs_ort_webgpu.py
```

该脚本会用仓库自带的 ORT 同时注册 CubeCL plugin 与官方 `onnxruntime-ep-webgpu`
plugin,逐 cell 做 CPU EP 参考对拍,并从 ORT profile 读回节点的实际 provider 归属
(node count 为 0 的 cell 标为 INVALID,而不是当成「很快」)。

## 与官方 WebGPU EP 的实测对比

见 [`docs/benchmarks/2026-08-20-cubecl-webgpu-vs-ort-webgpu.md`](../../docs/benchmarks/2026-08-20-cubecl-webgpu-vs-ort-webgpu.md)。
两条要点:

- **端到端每次 `Run` 比官方慢约 1.3–2.1x**,但这个差距**与工作量无关**(从
  add/small 到 add/large 工作量差几个数量级,比值不变),而 warmed profile 显示
  node dur 只有 8–9 us 而 wall 是 ~962 us。**优化目标是每次 Run 的固定路径,不是
  shader**。那张表几乎没测到 kernel。
- **f16 MatMul 精度优于官方**:我们与 ORT CPU 参考逐元素相等(`max_abs = 0`),
  官方最高 1.19。原因就是上面的 f32 累加 —— 这是一个用速度换精度的明确取舍。

## 正式来源

- [`onnx-runtime-ep-cubecl`](../../crates/onnx-runtime-ep-cubecl/src/lib.rs)
- [`backend.rs`](../../crates/onnx-runtime-ep-cubecl/src/backend.rs)
- [`memory.rs`](../../crates/onnx-runtime-ep-cubecl/src/memory.rs)
- [`runtime.rs`](../../crates/onnx-runtime-ep-cubecl/src/runtime.rs) — 含 f16 探测
- [`kernels/mod.rs`](../../crates/onnx-runtime-ep-cubecl/src/kernels/mod.rs) — dtype 分派与算子表
- [`provider.rs`](../../crates/onnx-runtime-ep-cubecl/src/provider.rs)
- [`onnx-runtime-ep-cubecl-plugin`](../../crates/onnx-runtime-ep-cubecl-plugin/src/lib.rs)
- [`ep_compat.rs`](../../crates/onnx-genai-ort/src/session/ep_compat.rs)

## 相关笔记

- [[execution/Execution Provider Contract]]
- [[execution/Plugin Execution Providers]]
- [[execution/Execution Backends]]
