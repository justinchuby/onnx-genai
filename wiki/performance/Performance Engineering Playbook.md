---
title: Performance Engineering Playbook
aliases:
  - Performance Measurement Discipline
  - 性能工程手册
tags:
  - performance
  - benchmarking
  - profiling
  - correctness
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# Performance Engineering Playbook

> [!summary] 本篇回答的问题
> 我们如何判断一个性能数字描述的是运行时本身、实验设置,还是一次测量错误?

一个 benchmark 不只是一个数字。它是关于某个特定模型、artifact、配置、机器和
workload 的一项断言。在解读结果之前,我们必须先确立产生这个结果的条件。

## 证据阶梯

按以下顺序采信证据:

1. **正确性输出** —— token ID 与必需的运行时不变量。
2. **确定性计数器** —— 字节数、page、capture、fallback 与各类记账。
3. **已同步的计时** —— 起止边界被明确理解的测量。
4. **profiler 归因** —— 算子、kernel 与设备级证据。
5. **wall-clock 吞吐** —— 在受控条件下重复得到的结果。

层级越高越容易解读。吞吐提升无法弥补 token 错误、graph capture 被禁用或内存
未被记账等问题。

## 先写下确切的断言

在运行实验之前先把断言写出来:

```text
On <platform>, for <model artifact>, at <context/batch/budget>,
change X should reduce counter Y without changing invariants Z.
```

这能暴露出缺失的控制项。“CUDA is faster”无法被检验;“fused GEMV 在 batch-1
decode 下减少了发出的指令数,同时保持 token ID 与 capture 不变”才可以被检验。

每一份上报的结果都应包含:

| 条件 | 示例 |
|---|---|
| Artifact | 模型 revision、量化、external-data 文件 |
| Runtime | commit、feature flag、provider 路径 |
| Workload | prompt、batch、context、warmup、生成的 token 数 |
| Hardware | 设备、OS/驱动模式、可用内存 |
| Isolation | 单独运行、每次采样前设备处于空闲 |
| Statistics | 采样次数、中位数与完整区间 |

## 检查实验是否强行决定了结果

一个被配置的旋钮可能直接决定了后来被解读为运行时发现的指标。例如,把最小 KV
bucket 设为等于最大容量,会使 `committed_len == capacity` 成立,即便正常运行时是
按需增长的。

要问:

> 如果系统什么有意思的事都没做,单凭这个旋钮本身会产生什么值?

所选的构造函数或入口点也可能是一个隐藏旋钮。像“stable address unavailable”这样
保守的默认值,可能意味着测试框架从未提供生产配置 —— 而不是硬件拒绝了该特性。

始终打印 decline predicate 的输入,而不仅仅是最终的 decline 原因。至少构建一个
能够证伪该断言的控制项。

## 检查仪器实际框住的是什么

GPU 操作通常是异步的:

```text
host starts timer
host enqueues copy
enqueue returns
host stops timer
GPU finishes copy later
```

那个计时器测量的是提交延迟(submission latency),而不是传输完成。用字节数除以
测得的时间:如果得到一个不可能的带宽,就是一个立即的警告 —— 说明该计数器被贴错
了标签,或者未做同步。

对每一个计时计数器,记录:

- 确切的起点与终点;
- 工作是否异步;
- 哪个 event、fence 或同步操作让完成变得可观测;
- profiler 开销是否改变了被测量的路径。

追踪架构与可用的 collector 参见 [[observability/Tracing and Profiling]]。

## 要求算术自洽

计算一个结果必须满足的界。如果测得的流量低于读取权重所需的理论最小值,应在断言
一项新优化之前先怀疑记账。

同样,避免在成员成本差异很大的总体上计算比率。一个 cache 可能报告了许多次小的
命中,却反复错过数 MB 的权重:

```text
count hit rate: looks healthy
byte hit rate: shows the real cost
```

选择与机器实际付出相匹配的单位:字节、page、指令或已同步的时间。

## 控制争用与不稳定的 wall-clock

共享的 GPU 活动会使同一份配置既产生假的回归,也产生假的收益。要在**每次采样之
前**确认设备处于空闲,而不是在一个长循环之前只确认一次。

在操作系统可以对 managed GPU 内存做分页的系统上,吞吐可能在没有代码改动的情况下
大幅波动。以进程本地的确定性计数器为主,把 wall-clock 当作辅助证据:

- 至少使用三个采样;
- 报告中位数与完整区间;
- 把计数器与吞吐并列呈现;
- 绝不反复选择性地重跑,直到出现想要的结果。

## 测量真实的 workload

一个方便的代理(proxy)只有在其差异被明确说明时才有用。顺序的设备拷贝无法复现
strided int4 GEMV 的访问模式;一个小 tensor 可能装得进 cache,而真实的工作集
装不下。

说明这个 proxy 省略了什么,当缓存可能改变结果时对尺寸做 sweep,并在真实路径被
测量之前把差距保留为未知。针对具体后端的示例,参见
[[execution/CPU Execution Provider]] 与
[[execution/CUDA Execution Provider]]。

## 证明被优化的路径确实运行了

一个没有可达性测试的加速实现,可能在每个单元测试都通过的同时仍然没有被接进
调用路径。一次完整的性能改动需要:

1. 针对该实现的正确性测试;
2. 一个证明 dispatch 会到达它的测试或计数器;
3. 针对不支持的设备或形状的 fallback 测试;
4. 在目标 workload 上的端到端测量。

测试必须自行构造其前置条件。不要依赖 allocator 恰好复用了某个地址,或某台机器
恰好暴露了某个特性;要非空泛地断言所需条件确实成立。

## 正确性约束高于速度

至少,相关改动应保持:

- 对相同 prompt 与配置产生逐字节相同的 token ID;
- graph capture 的预期,例如 `captures > 0` 与 `fallbacks == 0`;
- 内存边界与零超额分配(oversubscription);
- 零引用、字节与记账下溢;
- 安全析构 —— 尤其是 `Drop` 内不出现 panic/assertion。

一次更快但悄悄生成了更少或不同 token 的运行,是正确性失败,而不是性能收益。

## 上报一个结果

上报:

1. 断言与条件;
2. 正确性与确定性计数器;
3. 计时方法与采样;
4. 由 profiler 证据支撑的机制;
5. 可能收益的上限(ceiling);
6. 负面结果与尚未解决的不确定性。

如实的负面结果能避免重复劳动。来自某一个预算的小幅有利结果,不应自动变成全局
策略。在被取代的数字最初所在之处更正它们,以免后来的读者把它们当作事实引用。

## 相关笔记

- [[observability/Tracing and Profiling]]
- [[execution/Execution Backends]]
- [[execution/CPU Execution Provider]]
- [[execution/CUDA Execution Provider]]
- [[contracts/Runtime Contracts]]

## 正式来源

- [Measurement discipline skill](../../.github/skills/measurement-discipline/SKILL.md)
- [Kernel performance guide](../../docs/performance/KERNEL_PERF.md)
- [Benchmark artifacts](../../docs/benchmarks/)
- [Research notes](../../docs/research/)
