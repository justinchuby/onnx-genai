---
title: Documentation Guide
aliases:
  - Docs Reading Guide
  - Source Precedence
tags:
  - documentation
  - onboarding
  - evidence
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# Documentation Guide

> [!summary] 回答的问题
> 我应该相信哪份文档?面对相互重叠的设计、状态、研究与基准笔记,我该如何导航?

## 来源优先级

当各来源不一致时,按以下顺序采信:

1. 当前代码,加上可复现的测试或实测数据。
2. 针对该问题被明确指定为权威的文档。
3. 已接受的设计决策与当前的实现计划。
4. 与带日期修订绑定的状态笔记。
5. 研究/调查笔记。
6. wiki 解释性笔记。
7. 陈旧的 issue 描述与未经验证的 AI 对话。

“权威”指的是项目打算维护该文档,而不是说它永远不会出错。与之矛盾的实测数据
要求修正该文档。

## 目录含义

| 目录 | 用途 |
|---|---|
| `docs/architecture` | 项目/运行时结构与横切契约 |
| `docs/memory` | 内存证据、设计、VMM、KV 与 offload |
| `docs/execution` | EP、kernel、图捕获与放置 |
| `docs/genai` | 调度、流水线、元数据与模型包 |
| `docs/quantization` | 量化格式、kernel 与 MoE |
| `docs/performance` | 性能方法论与专项调查 |
| `docs/benchmarks` | 带条件、注明日期的实测记录 |
| `docs/ep-plugin` | 插件导出 ABI、缺口、安全与一致性 |
| `docs/distributed` | 通信、集合通信与多设备运行时 |
| `docs/status` | 当前进展与上游清单 |
| `docs/research` | 探索性工作;有用但不自动被采纳 |
| `wiki` | 学习路径、地图与通俗解释 |

## 按问题查找

| 问题 | 先读 |
|---|---|
| 产品架构是什么? | [`docs/architecture/DESIGN.md`](../../docs/architecture/DESIGN.md) |
| 原生运行时架构是什么? | [`docs/architecture/ORT2.md`](../../docs/architecture/ORT2.md) |
| 现在实现了哪些? | [`docs/status/PROGRESS.md`](../../docs/status/PROGRESS.md) 及代码 |
| 内存现在做了什么? | [`docs/memory/MEMORY_ARCHITECTURE.md`](../../docs/memory/MEMORY_ARCHITECTURE.md) |
| 提议中的内存模型是什么? | [`docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md`](../../docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md) |
| weight-offload 的北极星目标是什么? | [`docs/memory/WEIGHT_OFFLOAD.md`](../../docs/memory/WEIGHT_OFFLOAD.md) |
| CUDA 执行进展如何? | [`docs/execution/CUDA_EP_STATUS.md`](../../docs/execution/CUDA_EP_STATUS.md) |
| 实测数据显示了什么? | [`docs/benchmarks/README.md`](../../docs/benchmarks/README.md) 及某次带日期的运行 |

## 阅读带日期的证据

一份可信的性能笔记会写明:

- 代码修订版本;
- 模型与确切的产物;
- 硬件/驱动/平台;
- 后端与 EP;
- 精度与量化;
- batch、prompt/上下文与生成长度;
- 并发/竞争条件;
- 预热与重复方法;
- 什么变了、什么保持固定。

> [!important] 没有条件的数字不是结果
> 不要在缺少测量条件的情况下,把某个 tok/s、延迟或内存数字搬进决策。

## Wiki 与 docs 的分工

wiki 笔记应回答:

- “这个术语是什么意思?”
- “这些组件如何连接?”
- “我该从哪里开始?”
- “哪份正式来源掌握真相?”

正式文档应回答:

- “我们接受了什么契约?”
- “到底测量了什么?”
- “一个实现必须满足什么?”
- “当前的支持/状态矩阵是什么?”

如果一篇 wiki 笔记积累了规范性要求或基准证据,应把这些材料移入 `docs/`
并链接过去。

## 相关笔记

- [[start/Repository Map]]
- [[meta/Using this Wiki]]
- [[development/Testing and Verification]]
