---
title: API Design Principles
aliases:
  - Public API Design
  - Binding Design
tags:
  - api
  - architecture
  - ffi
  - compatibility
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# API Design Principles

> [!summary] 本篇回答的问题
> 一种行为如何在 Rust、CLI、HTTP、Python 与 C 之间保持一致,而不强迫每个表面都
> 暴露完全相同的机制?

## 分层表面

| 表面 | 主要受众 | 设计侧重 |
|---|---|---|
| Rust facade | Rust 应用 | 带类型的所有权与可组合性 |
| Engine API | 高级集成方 | 完整的生成策略/控制 |
| CLI/REPL | 人类与脚本 | 可发现性、稳定的 stdout、可操作的错误 |
| HTTP server | 兼容 OpenAI 的客户端 | wire 兼容、streaming、会话 |
| Python | Python/NumPy 用户 | 熟悉的 session API,显式处的 DLPack |
| C ABI | 跨语言 host | 不透明 handle、value/vtable ABI、显式所有权 |

这些表面共享的是语义,而不一定是完全相同的签名。

## 单一语义内核

prompt 处理、生成选项、finish reason、stop 行为与采样都应收敛到引擎的类型上。
binding 把数据 marshal 进这个内核,而不是重新实现生成策略。

例如,一个外部 sampler 替换的是终端 token 选择,而引擎仍然施加所配置的
processor/constraint 链。

## 可操作的失败

每个表面都应保留:

- 是哪个操作失败了;
- 涉及了哪个 argument/node/shape/dtype/path/device;
- 为什么被拒绝;
- 调用方如何修复;
- 在安全的前提下给出底层原因。

Rust 使用带类型的 crate 错误;编排层可以加上 `anyhow::Context`;C 使用机器可读的
状态码加上一条丰富的消息;Python/HTTP 在映射时不擦除诊断细节。

## 按语言划分的所有权

### Rust

利用所有权、borrowing 与 newtype 让非法状态难以表达。可失败的设备/运行时工作
返回 `Result`。

### C

- 不透明 handle 有配对的 create/release 函数;
- 在解引用之前检查 null;
- 每个 panic 都在 ABI 边界处被捕获;
- 状态/消息的所有权是显式的;
- 调用方与库绝不释放对方未指明的 heap 分配。

### Python

- 拷贝出的 NumPy 输出是安全的默认;
- DLPack/zero-copy 路径是显式的;
- thread/可重入行为被明确说明;
- iterator/callback 的完成语义必须区分 token event 与最终结果状态。

### HTTP

- request/response 的形状遵循目标中兼容 OpenAI 的契约;
- SSE streaming 保持 completion/error 语义;
- debug/admin 端点是 opt-in 的;
- 默认不记录 payload 与 secret。

## 兼容性策略

对仓库自有的 Rust/产品 API 而言,它处于 pre-release 阶段:干净地重塑它们并更新
所有调用方,而不是不断堆积别名与弃用。

这种自由不适用于:

- ONNX 语义/opset;
- 已文档化的模型 metadata;
- 稳定的 C/plugin ABI 版本;
- 向用户承诺的兼容 OpenAI 的 wire 行为;
- 受支持的 Python wheel ABI。

## 显式行为胜过隐藏的便利

- 除非显式启用,否则不做静默的 CPU 回退;
- eager API 中不做隐式的跨设备传输;
- 在使用可选行为之前先做 capability 协商;
- 在需要 metadata/图结构之处,不按模型名做 dispatch;
- 报告一个受约束的回退,而不是假装请求的模式已经运行。

## 正式来源

- [`RULES.md`](../../RULES.md)
- [`onnx-genai` facade](../../crates/onnx-genai/src/lib.rs)
- [`onnx-genai-server`](../../crates/onnx-genai-server/src/lib.rs)
- [`PYTHON.md`](../../docs/architecture/PYTHON.md)
- [`onnx-runtime-capi`](../../crates/onnx-runtime-capi/src/lib.rs)
- [`onnx-genai-capi`](../../crates/onnx-genai-capi/src/lib.rs)
- [`ERROR_AND_LOGGING_CONVENTIONS.md`](../../docs/architecture/ERROR_AND_LOGGING_CONVENTIONS.md)

## 相关笔记

- [[contracts/Runtime Contracts]]
- [[metadata/Metadata Driven Runtime]]
- [[execution/Plugin Execution Providers]]
