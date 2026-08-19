---
title: onnx-genai Knowledge Base
aliases:
  - Home
  - Wiki Home
tags:
  - wiki
  - home
status: maintained
lang: zh-CN
created: 2026-08-18
updated: 2026-08-19
---

# onnx-genai Knowledge Base

理解运行时是如何组织的、一个推理请求如何在其中流转,以及哪些契约让执行、
内存与插件保持正确。

本 wiki 以人为本来编写。每条路径都从一篇解释性笔记开始;指向源码与正式文档
的链接提供证据,而不会变成必须先做完的功课。

## 选择一条学习路径

### 理解本仓库

从 [[start/Repository Map]] 开始,然后依次:

1. [[architecture/Crate Architecture]]
2. [[architecture/Inference Request Lifecycle]]
3. [[execution/Execution Backends]]
4. [[execution/Execution Provider Contract]]

### 理解 execution provider

- [[execution/CPU Execution Provider]]
- [[execution/CUDA Execution Provider]]
- [[execution/Plugin Execution Providers]]

### 理解内存

[[memory/Memory Management for Beginners]] 从第一性原理讲解分配、虚拟后备
存储、共享映射、governor、holder,以及 provider 在 stream/context 上的职责。

[[memory/MoE Router Skew and Always-On Experts]] 讲解如何为一个训练过路由器的
MoE 模型测量专家选择的倾斜,以及为什么它决定了一项驻留策略是否有可利用之处。

[[memory/Virtual Memory for KV Cache]] 逐步讲解当 KV cache 增长、当前缀
被共享、当一个物理句柄被映射到多个虚拟地址、以及当 release 或拆除同步失败时,
虚拟内存管理器实际做了什么。

### 理解模型提示词格式

[[prompting/Chat Templates]] 讲解一个消息数组如何变成模型看到的 token
序列:模板结构与约定、为什么生成从渲染出的 assistant 前缀之后的位置开始、
多模态占位符的变体、按接收方限定作用域的 channel(`to=self` / `to=user` /
`to=<tool>`),以及工具调用如何被渲染并回喂。

[[performance/Chunked Prefill]] 解释了为什么一段 prompt 可以分块 prefill,而不必
重算某一块所要 attend 的 KV;限制 query 宽度换来了什么;以及为什么最后那个残余
块必须补齐到一个固定的阶梯上,kernel cache 才能不再在每次请求时重新编译图。

### 修改或测量运行时

- [[development/Testing and Verification]]
- [[performance/Performance Engineering Playbook]]
- [[performance/Chunked Prefill]]
- [[observability/Tracing and Profiling]]
- [[api/API Design Principles]]

### 理解契约与模型

- [[contracts/Runtime Contracts]]
- [[contracts/Formal Verification with TLA+]]
- [[metadata/Metadata Driven Runtime]]
- [[metadata/Model Packages and Variants]]

## 来源优先级

> [!important] 解释不是规范
> 当前代码与可复现的测量优先,其次是 `docs/` 下的权威文档、已接受的设计决策,
> 最后才是这些解释性笔记。

本 wiki 区分已发行的行为与提议中的设计。如果一篇笔记与实现不一致,请将其视为
文档 bug,并核对当前源码。

## 在 Obsidian 中阅读

完整的 `wiki/` 目录是一个兼容 Obsidian 的 vault。笔记使用稳定的英文路径、
YAML 属性、wikilink 与 callout,同时在 GitHub 和发布后的站点上仍是普通
Markdown。

参见 [[README|Wiki index and conventions]] 查看每一篇笔记,以及
[[meta/Using this Wiki]] 了解写作细节。
