---
title: Runtime Contracts
aliases:
  - Contract Map
  - Runtime Invariants
tags:
  - contracts
  - architecture
  - invariants
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# Runtime Contracts

> [!summary] 本篇回答的问题
> 哪些跨层承诺能防止各自看起来合理的组件被组合成一个不正确的运行时?

契约(contract)陈述的是一层可以对另一层做出哪些假设。当它被编码进类型、测试、
capability token 或 ABI 结构中时,它比一条代码注释更强。

## 契约家族

| 契约 | 生产方 | 消费方 |
|---|---|---|
| 模型语义 | metadata/package/exporter | engine、loader、scheduler、backend |
| 图语义 | loader/IR/形状推理 | optimizer、placement、EP |
| EP claim | EP capability/compile | session executor |
| Tensor 所有权 | allocator/session/EP | kernel 与 binding |
| 异步顺序 | EP stream/fence | executor 与释放路径 |
| 内存容量 | authority/Governor | 持有者与机制 |
| 持久状态 | model/engine/backend | checkpoint、fork、migrate、resume |
| Plugin ABI | ABI crate/host | 动态 provider |
| 公共 API | engine | CLI/server/Python/C 客户端 |

## 设计规则

### 让非法状态无法表示

使用:

- 为 ID、长度、offset 以及 token/page 计数使用 newtype;
- 用 owned handle 表示所有权;
- 用 borrow 表示临时视图;
- 用 capability 值表示已解析的可选行为;
- 用 enum 表示显式的状态转换。

测试应当验证行为,而不是去弥补一个允许任意跨设备或跨所有者组合的 API。

### 先预留,再生效

对于会改变可见状态的内存变更:

```text
plan → reserve → provisional execute → commit
```

commit 之前的失败会恢复旧状态并归还预留的容量。等待不得在持有部分稀缺资源或
治理锁(governance lock)时发生。

### 由持有者选择被牺牲的对象

authority 可以请求归还字节;但它无法安全地判定要删除哪个权重、KV page 或
in-flight buffer。策略应放在理解 pinning、重算成本与执行状态的持有者一侧。

### claim 是诚实的

一个 EP 或后端的 claim 是一项承诺:被编译出的路径支持已解析的 opset、shape、
dtype、layout 与所需 capability。“Unsupported”是一个有效的、可诊断的结果;
悄无声息的语义回退不是。

### 状态一起提交

生成状态可能包含 KV、recurrent/conv 状态、sampler/search 状态与请求进度。一次
migration/checkpoint/step 绝不能暴露出旧组件与新组件的混合体。

### ABI 显式地拥有生命周期

每一个跨模块指针都需要:

- 唯一的 owner;
- 在正确模块中的 release 操作;
- 一个有效期区间;
- version/layout 协商;
- panic 与 error 的隔离;
- 当回调或对象仍存在时对 unload 的 pinning。

## 契约如何被强制执行

1. Rust 类型系统与 borrowing。
2. 构造函数校验。
3. Capability 协商。
4. 聚焦的不变量测试。
5. 端到端的一致性/parity 测试。
6. 暴露下溢/未记账字节的运行时计数器。
7. 带版本的 ABI 记录。
8. 在最早的边界处给出可操作的失败。
9. 针对并发协议的 model checking 加上 trace refinement。

> [!warning] 构建通过不等于集成证据
> 两个组件可以各自针对重复或彼此脱节的契约通过编译,却从不互相驱动对方。工作区
> 成员关系、round-trip 测试与唯一权威的契约定义,都是正确性的一部分。

## 正式来源

- [`RULES.md`](../../RULES.md)
- [`MEMORY_MANAGEMENT_MODEL_DESIGN.md`](../../docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md)
- [`NXRT_ABI.md`](../../docs/architecture/NXRT_ABI.md)
- [`EP_CONFORMANCE.md`](../../docs/execution/EP_CONFORMANCE.md)
- [`MODEL_METADATA.md`](../../docs/genai/MODEL_METADATA.md)
- [TLA+ model index](../../specs/tla/README.md)

## 相关笔记

- [[execution/Execution Provider Contract]]
- [[memory/Memory Management for Beginners]]
- [[metadata/Metadata Driven Runtime]]
- [[execution/Plugin Execution Providers]]
- [[contracts/Formal Verification with TLA+]]
