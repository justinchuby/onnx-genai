---
title: Metadata Driven Runtime
aliases:
  - Inference Metadata
  - Model Agnostic Runtime
tags:
  - metadata
  - models
  - contracts
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# Metadata Driven Runtime

> [!summary] 本文回答的问题
> 运行时如何在不硬编码模型名称和维度的情况下支持不同的模型架构?

运行时把模型行为当作数据来对待。推理元数据、图结构、ONNX 语义、能力
(capability)以及显式的用户配置,取代了按模型族(model-family)分派的做法。

## 元数据描述了什么

取决于具体的包/模型,元数据可以声明:

- 模型组件与流水线拓扑;
- token/embedding 输入与 logits/hidden 输出;
- KV 与递归状态(recurrent state)的所有权;
- attention/KV 的几何结构与轴;
- tokenizer/chat-template 行为;
- 图像/音频预处理程序;
- 生成默认值与约束;
- 投机式(speculative)proposer 契约;
- 运行时所需的能力;
- 执行/放置提示;
- 量化与硬件要求。

## 能力协商

模型列出稳定的能力标识符。运行时公布其支持的集合,并在缺失能力时明确地
加载失败,而不是从模型身份去猜测。

```text
model requires: [loop_carried_state, multi_axis_positions]
runtime offers: [loop_carried_state]
result: fail clearly; missing multi_axis_positions
```

能力字符串是数据,不应按模型名称变成隐藏的分支。

## 解析优先级

对于执行提示,更强/更局部的用户意图会覆盖内嵌的默认值:

1. 编程式的 builder API;
2. 用户的 `execution_hints.json`;
3. 推理元数据中的执行提示;
4. ONNX 的 `onnx_runtime.*` 元数据属性。

相互冲突的强制约束应当失败,而不是悄悄地做出选择。

## 名称与角色

Tensor 端口名称是数据,而非语义。解析应优先:

1. 精确声明的角色/端口;
2. 唯一的结构/形状信号;
3. 在有歧义时,以指明所需元数据的方式失败。

针对狭窄的兼容性场景,可能存在按惯用名称的回退,但已声明的契约始终优先,
且该回退不得在有歧义的端口之间猜测。

## 状态所有权

元数据区分:

- 作为序列来源的 token ID 与 输入 embedding;
- 自有(owned)与共享(shared)的 KV;
- 位置相关的 KV 输入/输出;
- hidden/recurrent 输出;
- proposer 到 target 的共享状态映射。

这些声明让 ORT 和原生后端能够实现相同的生成语义,而无需把模型族的假设复制
进引擎代码中。

## Schema 与前向演进

`onnx-genai-metadata` 提供带类型的 Rust 结构、校验以及确定性的 JSON Schema
生成。读取方应当:

- 校验所需的能力;
- 对无效字段保留可操作的路径;
- 定义 schema 版本行为;
- 容忍显式允许的加性(additive)演进;
- 拒绝会改变语义的歧义。

## 元数据不是性能许愿单

提示不会覆盖正确性、物理能力或用户策略。对 CUDA graph capture、某个 kernel
或某个 device 的请求,仍然要通过能力与兼容性检查。

## 形式化来源

- [`onnx-genai-metadata`](../../crates/onnx-genai-metadata/src/lib.rs)
- [`MODEL_METADATA.md`](../../docs/genai/MODEL_METADATA.md)
- [`MODEL_PACKAGE.md`](../../docs/genai/MODEL_PACKAGE.md)
- [`RULES.md`](../../RULES.md)

## 相关笔记

- [[metadata/Model Packages and Variants]]
- [[contracts/Runtime Contracts]]
- [[api/API Design Principles]]
- [[architecture/Inference Request Lifecycle]]
