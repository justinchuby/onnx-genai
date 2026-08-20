---
title: Execution Backends
aliases:
  - ORT and Native Backends
  - Backend Selection
tags:
  - execution
  - onnx-runtime
  - nxrt
  - architecture
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-20
---

# Execution Backends

> [!summary] 回答的问题
> ORT 后端与原生(native)后端有什么区别?execution provider 又落在哪里?

## 三个容易混淆的概念

| 概念 | 含义 |
|---|---|
| Generation engine | 拥有 prompt/decode/sampling/session 语义 |
| Decode backend | 通过 ORT 或原生 nxrt 执行模型的单步 |
| Execution Provider(执行提供者,EP) | 为某类设备(如 CPU、CUDA 或 CubeCL)实现/认领图算子 |

后端(backend)不等于设备(device)。原生 CUDA 指的是“使用 CUDA EP 的 nxrt
后端”;ORT CUDA 指的是“配置了其 CUDA provider 的 ONNX Runtime 后端”。

## ORT 后端

ORT 路径通过 `onnx-genai-ort` 使用 ONNX Runtime session。

其优势包括:

- 成熟的算子与模型覆盖;
- 现有的 ORT execution-provider 生态;
- 强有力的对齐/参考路径;
- 成熟的图优化与运行时行为。

generation engine 仍然拥有外层的 token 循环、调度、sampling 与 session 语义;
ORT 负责图执行。

## 原生 nxrt 后端

原生路径使用 `onnx-runtime-session` 与 `onnx-runtime-*` 技术栈:

```text
ONNX/model package
    ↓
loader → IR → shape inference → optimization
    ↓
placement / EP claims / kernel compilation
    ↓
memory plan + executor
    ↓
CPU、CUDA、CubeCL 或 plugin EP kernels
```

它对以下方面提供直接控制:

- 图 IR 与 layout;
- kernel 选择与放置;
- activation planning;
- 设备传输与 fence;
- CUDA graph capture;
- VMM 支撑的分配;
- 追踪与分阶段 profiling。

原生并不自动意味着更快。覆盖度、kernel 质量、图形状、硬件、capture 状态与内存
策略都会产生影响。

## Execution provider

`onnx-runtime-ep-api` 定义了原生 EP/kernel 契约。具体的 provider 包括 CPU、CUDA 与
CubeCL。插件 crate 桥接外部/动态 provider。

一个 EP 通常拥有或提供:

- 设备身份与能力;
- 算子支持与 kernel factory;
- 设备 context 与 stream;
- 分配/拷贝/提交机制;
- graph-capture 行为;
- 同步与释放顺序。

它不应拥有 generation policy、请求优先级或全局逐出(eviction)选择。

## 后端/设备选择

CLI 支持分别选择后端与 EP。更换模型、后端或 EP 需要重新加载,因为 session 是针对
某种图执行策略构建的,无法简单地把它的活动状态迁移到另一个 provider。

在编译/可用的前提下,环境变量与 CLI 配置可以选择 CPU、CUDA、WebGPU、CubeCL、CoreML 或
插件 provider。请求了不可用的 provider 会明确失败,除非显式启用了 fallback。

## 为什么两条路径都保留

1. **对齐:** ORT 为原生正确性调查提供参考。
2. **覆盖:** 原生不支持的算子仍可通过 ORT 或异构/插件路径获得。
3. **性能实验:** nxrt 暴露 planner、kernel 与内存选择。
4. **生态兼容:** 插件 EP 工作保护了对既有硬件后端的访问。
5. **渐进迁移:** generation 特性可保持后端无关。

## 正确地比较性能

> [!warning] 只看后端标签是不够的
> 只有在模型、精度、EP/设备、batch、上下文、内存策略、graph capture、线程与系统
> 竞争都被固定并报告时,“原生 vs ORT”才有意义。

从这里开始:

- [`docs/benchmarks/README.md`](../../docs/benchmarks/README.md)
- [`docs/execution/NATIVE_CUDA_DECODE.md`](../../docs/execution/NATIVE_CUDA_DECODE.md)
- [`docs/execution/CUDA_EP_STATUS.md`](../../docs/execution/CUDA_EP_STATUS.md)
- [`docs/execution/EP_CONFORMANCE.md`](../../docs/execution/EP_CONFORMANCE.md)

## 相关笔记

- [[architecture/Crate Architecture]]
- [[architecture/Inference Request Lifecycle]]
- [[execution/CubeCL Execution Providers]]
- [[memory/Memory Management for Beginners]]
