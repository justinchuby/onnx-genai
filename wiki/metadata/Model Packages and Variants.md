---
title: Model Packages and Variants
aliases:
  - Model Package
  - Package Variant Selection
tags:
  - metadata
  - model-package
  - deployment
status: proposed
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# Model Packages and Variants

> [!summary] 回答的问题
> 为什么 model package 不只是一个装着 ONNX 文件的目录?运行时又该如何安全地选择
> 特定于硬件的变体(variant)?

> [!warning] 提案状态
> 正式的 model-package 文档是一份提案,并明确表示这并不意味着已完整实现。在依赖
> 某条命令或某个格式特性之前,请先核实当前的工具链。

## 部署问题

一个可部署的 GenAI 模型可能包含:

- 一个或多个 ONNX 图;
- 外部权重;
- tokenizer 与 chat template;
- 推理元数据或兼容性配置;
- 图像/音频 processor;
- adapter 与 speculative draft 模型;
- 已编译的 EP context;
- 多个硬件/设备变体。

松散的相对路径无法提供共同的身份标识、完整性清单,也无法确定性地解释到底运行了
哪个变体。

## 打包目标

1. 可移植、离线的分发。
2. 可复现的图/权重/tokenizer/编译器身份标识。
3. 与 zero-copy 兼容的外部权重。
4. 同一逻辑模型下的特定硬件变体。
5. 已编译 EP context 的复用。
6. 可检视的校验与选择过程。
7. 明确的信任边界与路径限定。

## 概念布局

```text
package_root/
├── manifest.json
├── decoder/
│   ├── component.json
│   ├── cpu/
│   └── cuda/
└── shared_assets/
    └── sha256-<digest>/
        ├── tokenizer.json
        └── chat_template.jinja
```

从概念上讲,package 是一个带有 manifest、组件、变体与内容寻址(content-addressed)
共享资产的目录——不一定是一个压缩归档。

## 变体选择

一个变体可以声明:

- EP/provider 身份;
- 设备类别;
- 兼容性字符串;
- executor 专属信息;
- model/context/external-data 路径。

选择过程应当:

1. 按请求的/可用的 EP 与设备过滤;
2. 请求 provider 校验不透明的已编译兼容性;
3. 确定性地排序;
4. 解释被选中/被拒绝的候选;
5. 绝不静默地运行哈希/兼容性不匹配的情况。

## 信任与完整性

- 可移植布局始终被限定在 package root 内。
- 已安装布局需要显式的主机信任。
- symlink 与路径穿越需要有意为之的策略。
- 共享资产使用内容寻址身份标识。
- 哈希同时覆盖文件名与字节内容。
- 加载不应隐式抓取缺失的内容。

## 与推理元数据的关系

package 回答的是**产物在哪里、哪个变体是兼容的**。推理元数据回答的是**模型意味着
什么、它需要哪些运行时能力**。package 可以携带元数据,但它不替代语义校验。

## 正式来源

- [`MODEL_PACKAGE.md`](../../docs/genai/MODEL_PACKAGE.md)
- [`MODEL_METADATA.md`](../../docs/genai/MODEL_METADATA.md)
- [`onnx-model-package`](../../crates/onnx-model-package/src/lib.rs)
- [[metadata/Metadata Driven Runtime]]
- [[architecture/Inference Request Lifecycle]]
- [[contracts/Runtime Contracts]]
