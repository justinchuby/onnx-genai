---
title: Plugin Execution Providers
aliases:
  - Plugin EPs
  - EP ABI
tags:
  - execution
  - ep
  - plugin
  - abi
  - ffi
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-20
---

# Plugin Execution Providers

> [!summary] 回答的问题
> execution provider 如何跨越动态库边界,而不至于混淆 Rust trait、ORT 的 plugin ABI 与 nxrt 的原生 ABI?

本仓库支持三个相关的表面:

| 表面 | 用途 |
|---|---|
| Rust `ExecutionProvider` trait | 进程内的原生 nxrt EP 契约 |
| ORT plugin-EP C ABI 适配器 | 导出/加载与 ORT plugin 协议兼容的 provider |
| 原生 nxrt 动态 ABI | 把树外(out-of-tree)EP 直接加载进 nxrt |

它们应当表达相容的能力,但并不是同一个 ABI。

## 为什么需要一个 C ABI

Rust 的 trait-object 布局与 `Arc` 所有权都不是稳定的动态库契约。一个 plugin 边界
需要:

- `#[repr(C)]` 的 value/vtable 布局;
- 显式的版本与大小协商;
- 每个指针都有清晰的所有权;
- panic 的隔离;
- allocator/module 安全的状态传递;
- 显式的可选回调。

host 在验证之后,把这个 ABI 包装为安全的 Rust 类型。

## 单一事实来源

> [!important] 消费方依赖于 ABI crate
> host 与 plugin 必须导入同一份契约定义。重新声明一份私有副本可能照样编译成功,
> 而符号名、布局或所有权规则却互相不一致。

原生 host 依赖 `onnx-runtime-ep-nxrt-abi`;测试用的 plugin 是真实的 workspace
成员,因此常规检查不会意外地把它们漏掉。

## ORT plugin 导出

ORT 加载一组众所周知的导出符号,包括 `CreateEpFactories` 和 `ReleaseEpFactory`,
随后通过 C vtable 与 factory、device、allocator、transfer、claim 和 compute 交互。

关键规则:

- 从真实的 header/运行时行为解析名称,而不是靠 typedef 猜测;
- 协商 ORT API 版本;
- 为每个 `extern "C"` 回调设置针对 panic 的栅栏;
- 让 ORT 拥有的/回调帧内的指针保持在其文档规定的生命周期内;
- 对不支持的 transfer/capability 方向 fail closed;
- 保持 Rust-trait 与 C-ABI 的 claim 一致。

## 所有权教训

1. **唯一有文档记载的 owner。** 若 ORT 保留了某个指针,即便注册调用已返回,立即
   释放它也是一次 use-after-free。
2. **回调生命周期是一条硬边界。** kernel-context 指针不能被存留到 compute 之后。
3. **不跨模块的堆所有权。** 在某个 CRT/模块中分配的数据不应由另一个来释放。使用
   内联/固定的状态值,或同模块的 release 回调,可以避免这一点。
4. **卸载是一个生命周期事件。** 只要仍有 handle、回调、kernel、分配或延迟释放引用
   着某模块的代码,该模块就不能卸载。

## Trait/ABI 一致性

C ABI 掌握的信息可能少于原生 trait,尤其是在形状推断方面。一个 plugin 通过 C
声明的能力,不得超过它能安全编译的范围:

```text
C claims = native trait claims ∩ nodes resolvable through the ABI
```

parity 测试应同时覆盖肯定的情形与刻意拒绝(decline)的情形。

## 外部 provider

CPU 和 CUDA 有原生的树内(in-tree)实现。产品配置暴露的其他 provider 名称——例如
WebGPU、CoreML、QNN、OpenVINO 或 CubeCL 的 `cubecl-webgpu`/`cubecl-vulkan`——
在没有对应代码存在时,不得被描述为原生的树内实现。它们一般通过 ONNX Runtime
或 plugin 路径抵达,并继承那些路径的 host/版本/平台约束。CubeCL plugin 是本仓库
提供的双 ABI cdylib,但仍然通过同一 plugin 加载纪律进入 ORT/nxrt。

## 正式来源

- [`NXRT_ABI.md`](../../docs/architecture/NXRT_ABI.md)
- [`EP_PLUGIN_EXPORT_ABI_TRUTH.md`](../../docs/ep-plugin/EP_PLUGIN_EXPORT_ABI_TRUTH.md)
- [`onnx-runtime-ep-plugin`](../../crates/onnx-runtime-ep-plugin/src/lib.rs)
- [`onnx-runtime-ep-nxrt-abi`](../../crates/onnx-runtime-ep-nxrt-abi/src/lib.rs)
- [`onnx-runtime-ep-nxrt-host`](../../crates/onnx-runtime-ep-nxrt-host/src/lib.rs)
- [`onnx-runtime-ep-cubecl-plugin`](../../crates/onnx-runtime-ep-cubecl-plugin/src/lib.rs)

## 相关笔记

- [[execution/Execution Provider Contract]]
- [[execution/CubeCL Execution Providers]]
- [[contracts/Runtime Contracts]]
- [[api/API Design Principles]]
