---
title: CPU Execution Provider
aliases:
  - CPU EP
tags:
  - execution
  - cpu
  - ep
  - performance
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# CPU Execution Provider

> [!summary] 本文回答的问题
> 原生 CPU EP 如何在可移植性、正确性、SIMD 性能、线程与内存之间取得平衡?

CPU EP 既是一个可移植的执行后端,也是原生运行时最易获取的正确性基线。它实现
了共享的 [[execution/Execution Provider Contract]],并按 ONNX 的 domain/op
type 注册 kernel。

## 后端策略

```text
ONNX node
  ↓ registry + shape/dtype checks
CPU kernel
  ├── portable Generic/reference path
  ├── built-in x86 SIMD path when available
  └── MLAS-backed paths for selected operations/features
```

即便存在优化路径,可移植路径依然重要:

- 它无需厂商 toolkit 即可运行;
- 它提供了一个正确性对照;
- 它避免了缺失某个 ISA 变成运行时失败;
- 它为测试提供了隔离优化路径 bug 的简单手段。

## 运行时能力,而非构建机器身份

快速路径应当依据运行时的 CPU 能力和 tensor 需求来选择。AVX-512/AVX2/NEON/SVE
是否可用改变的是速度,而非语义。不受支持的指令必须降级到一条正确的路径。

Kernel 由 shape 和 dtype 驱动。模型名称和固定的 hidden 维度不应出现在 EP 中。

## 热路径架构

CPU EP 包含:

- 分块/寄存器分片(register-tiled)的 GEMM 与 SIMD 后端;
- 量化 matmul 与 MoE kernel;
- attention、normalization、索引与数据搬移 kernel;
- EP 专属的融合/优化 pass;
- host 并行、decode 亲和性与 NUMA 感知支持;
- 权重卸载(offload)放置与 host-cache 机制。

session 应当观察到一个 `Kernel`,而不是内部运行了哪一种 GEMM 实现。

## 线程方面的经验

线程数是算法的一部分:

- 每线程的 scratch 乘以 worker 数量,可能变成进程规模的内存占用;
- 嵌套并行可能导致核心超额订阅(oversubscribe);
- decode 通常受益于一个有界的 worker pool,而非所有可用的硬件线程;
- 对大权重而言,NUMA 放置可能压过算术层面的改进;
- 进程亲和性与运行时线程预算是不同的控制项。

任何随模型权重或线程数增长的、常驻的每线程或每 kernel 缓冲区,都必须以实际
字节数规划,并且可被拒绝(declinable)。

## 持久缓存

CPU 性能可能会使用:

- 转置后的权重缓存;
- 稠密/加宽的权重缓存;
- 量化的 packed-B 缓冲区;
- 常驻的反量化权重;
- 可复用的大块 host 分配;
- accumulator scratch 池。

> [!warning] 缓存就是一种内存策略
> 如果它的存活超过一次 kernel 调用,并且随权重或线程数增长,那么它必须在分配
> 之前声明、按实际占用记账,并在被拒绝时有正确的回退。

## 正确性与性能门槛

一次优化应当保持:

- 输出值在有正当理由的容差范围内;
- 在预期确定性生成之处,token ID 逐字节一致;
- 受支持的 shape/dtype/opset;
- 显式的回退行为;
- 有界的持久内存;
- 对没有快速 ISA 的机器的可移植性。

kernel 对旧 kernel 的加速不足以作为证据。请以生产环境的 shape 对比相关的
ORT CPU EP 或另一个强基线。

## 形式化来源

- [`onnx-runtime-ep-cpu`](../../crates/onnx-runtime-ep-cpu/src/lib.rs)
- [`docs/performance/KERNEL_PERF.md`](../../docs/performance/KERNEL_PERF.md)
- [`docs/performance/CPU_MATMUL_ASSIGNMENT.md`](../../docs/performance/CPU_MATMUL_ASSIGNMENT.md)
- [`CPU EP vs ORT benchmark`](../../docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md)
- [`docs/architecture/CROSS_PLATFORM.md`](../../docs/architecture/CROSS_PLATFORM.md)

## 相关笔记

- [[execution/Execution Provider Contract]]
- [[performance/Performance Engineering Playbook]]
- [[memory/Memory Management for Beginners]]
