---
title: MoE Router Skew and Always-On Experts
aliases:
  - Always-On Experts
  - Expert Selection Skew
  - How MoE Router Skew Was Measured
tags:
  - memory
  - moe
  - measurement
  - offload
  - beginner
status: maintained
lang: zh-CN
created: 2026-08-18
updated: 2026-08-19
---

# MoE Router Skew and Always-On Experts

> [!summary] 回答的问题
> 在 Mixture-of-Experts 模型中,router 是否会挑选某些 expert 的频率远高于其他
> expert——高到把热门 expert 常驻(resident)就能带来收益——**而你又如何诚实地测量
> 这一点**?本笔记解释了那个实测结果背后的方法:`granite-3.0-1b-a400m` 存在
> **always-on expert** 且带有沉重的选择长尾。这些数字及其复现记录在 benchmark
> 记录中;本笔记解释它们的含义,以及如何不自欺。

## 这个问题为什么存在

一个 dense decoder 在每一步都**恰好读取每个权重一次**。内存设计把这一点记录为在全部
867 个权重键上 `reads_per_step = 1.000`——不存在“热”子集,因此任何 residency 策略都
无法偏好某个权重:把权重 A 常驻而不是 B,并不能省下什么,因为两者每一步都会被读取。
这就是为什么 dense 权重流式传输被定论为“非收益项”。

Mixture-of-Experts(MoE)层则不同。每一层有许多 expert(小型前馈网络),但一个很小的
**router** 每个 token 只挑选其中 `k` 个。如果 router 的选择均匀分布,MoE 在 residency
上就不会比 dense 更好:每个 expert 大约在 `k / E` 的时间被读取,没有哪个值得固定
(pin)。但如果 router **发生倾斜(skew)**——一个 token 接一个 token 地回到同样那几个
expert——那么这些 expert 就真的被反复读取,把它们常驻而把其余 expert 换页(page)就可能
带来收益。内存设计称 MoE 是“residency 策略第一次真正有对错可言的情形”,并要求在设计任何
策略**之前**先对此进行**测量**。关于周边的 allocator/backing/governor 术语,见
[[memory/Memory Management for Beginners]]。

## “always-on expert”的精确含义

> [!important] 定义
> **always-on expert** 是指在其所在层的**100% 的 decode 步骤中都被选中**的 expert——
> 不只是一个频繁的 expert,而是 router 从不跳过的那个。它是可能存在的最强复用形式:
> 这样的 expert 可以在**零**预测的情况下被常驻固定,因为它每一步都需要。

在被测模型中,这一现象出现在**第 1 层与第 2 层**。每一层都有*某个*最热的 expert
(被选中的比例为 46–100%),但只有部分层拥有一个被选中比例达 100% 的 expert。

> [!warning] 这是被测模型的性质,而非 MoE 的定律
> “第 1–2 层是 always-on”对 `granite-3.0-1b-a400m` 在所测 prompt 上成立。它**不是**关于
> MoE 模型的普遍事实。换一个 checkpoint 可能把其 always-on expert 放在别处,或者根本没有。
> 该测量确立的是倾斜*可以*强到足以利用,而不是它总是如此。

## 模型,以及一个会毁掉测量的陷阱

测量使用 `granite-3.0-1b-a400m-instruct`:32 个 expert、top-8 路由、24 层、无 shared
expert,通过 Mobius 导出为 **f16 dense**。这里的“dense”指的是 ONNX 图是对全部 32 个
expert 展开的一个循环,带有每层的 `TopK`,而这正是让每一层的 expert 选择作为图输出可被
观测的原因。

> [!danger] 最重要的单条方法论要点
> **router 必须是真正训练过的。** 倾斜是*训练过的* router 权重的性质。一个随机初始化的
> router 会**从构造上均匀地**选择 expert,因此它会测出一个平坦的 `reads_per_step ≈ k/E`,
> 并给出一个**自信的假阴性**——“MoE 没有复用,停止这项工作”——而这是随机权重的假象,并非
> 关于 MoE 的事实。这是下一个人最可能掉进去的陷阱。因此测量导出的是一个真实的 IBM Granite
> checkpoint;它绝不合成一个带随机权重的玩具 MoE。如果你拿不到一个训练过的 router,就无法
> 回答这个问题——一个合成 router 的“无倾斜”什么都说明不了。

## 为什么在 CPU 上测量是有效的

expert 选择是 `indices = TopK(MatMul(hidden, gate_weight), k)`。被选中的 expert 集合是
对一个矩阵乘积取整数 top-`k`;它在 f16 与 f32 之间、在 CPU 与 CUDA execution provider
之间都**不会**改变。所以*哪些 expert*这个问题与 dtype 和 EP 无关,CPU 的选择与 CUDA 的
完全相同。这正是允许在 [[execution/CPU Execution Provider]] 上测量倾斜、并把结论应用到
CUDA decode 的依据。只有*时序与带宽*才依赖 EP——而那些是单独测量的,不是从这条 trace
推断出来的。

## 采样设计,以及它所防御的反驳

这条 trace 对 **3 个 prompt**——英文散文、Python 代码与数学——各解码 **64 个 greedy
token**(合计 192 个 decode 步骤),外加每个 prompt 的 prefill。

- **三个内容领域**防止单一主题制造出私有的热集合:如果倾斜只在散文中出现,那它可能关乎那段
  文本,而非 router。
- **prefill 被单独分析。** greedy 解码可能陷入循环,因此只在 decode 中看到的热集合可能是
  模型自我重复的假象。测量 **prefill** token——它们是多样、外部提供的 prompt,而非模型
  输出——就是在真正多变的输入上检验倾斜。倾斜在那里依然存在(top-8 占比 ≈ 0.49–0.55),
  这正是对那个显而易见的反驳的回答。

## 解读统计量:看形状,而非均值

> [!important] 均值被固定在 `k/E`;只有分布携带信息
> 因为每一层总是恰好从 `E` 个 expert 中选出 `k` 个,所有 expert 上的平均
> `reads_per_step` **从构造上就是 `k/E = 8/32 = 0.250`**。报告均值什么也说明不了。整个
> 问题在于分布的**形状**:平坦意味着无复用(停止),沉重的长尾意味着可利用的复用(继续)。

实测的形状是一条沉重的长尾:中位数接近均匀值 `0.229`,**最大值为 1.000**(always-on
expert),32 个 expert 中的 top-8 承载了 **45.4%** 的读取量(对比 25% 的均匀基线),
Gini 为 0.334。具体地说:一个固定热门 expert 的 residency 策略会获得真实的命中,而在均匀
路由下它做不到。

## 这允许了什么——以及不允许什么

> [!note] 这回答了一个问题,并开启了下一个
> - **它表明** residency 策略确有真实可利用的东西——内存设计悬而未决的 MoE 问题得到了
>   肯定的回答。always-on expert 是免费的、零预测的固定项。
> - **它并未表明某个策略就会取胜。** 如今换页层把整个 expert bank 当作**一个键**换页,
>   因此倾斜对运行时是*不可见*的(它报告整个 bank 的 `reads_per_step ≈ 1.0`,类似 dense)。
>   而且 per-expert 的 VMM 换页是一种**大 expert**技术——2 MiB 的设备 granule 使得
>   granite 的亚 granule int4 expert 无法被单独换页。这两条告诫都在 per-expert 换页 churn
>   记录中被测量。

因此,管道搭建(per-expert 换页,使倾斜*可见*)先于策略(利用它)。关于这个先后次序及其实测
成本,见所链接的 churn benchmark。

## 复现

确切的调用方式、环境与原始计数都在 benchmark 记录中。简而言之,从仓库根目录:

```powershell
python scripts/moe_router_skew.py
```

这会把每一层的 `TopK` indices 打补丁成一个图输出,在 CPU EP 上 greedy 解码这三个 prompt,
打印每个 prompt 与聚合的表格,并写出 `scripts/moe_router_skew_counts.json`——表格据以计算的、
已提交的原始计数。该运行是确定性的,因此它会精确复现已提交的 JSON。关于添加图输出与读取
per-op 信号的机制,见 [[observability/Tracing and Profiling]]。

## 正式来源

- 实测证据(数字、硬件、方法、house rule §32.2):
  [Router skew benchmark](../../docs/benchmarks/2026-08-18-moe-router-skew-granite.md)
- 换页成本、granule 下限,以及为什么倾斜如今不可见:
  [Per-expert MoE paging churn](../../docs/benchmarks/2026-08-18-moe-per-expert-paging-churn.md)
- 这所回答的悬而未决的问题:
  [Memory Management Model Design](../../docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md)
- 复现脚本与原始计数:
  [`scripts/moe_router_skew.py`](../../scripts/moe_router_skew.py),
  [`scripts/moe_router_skew_counts.json`](../../scripts/moe_router_skew_counts.json)
- 相关笔记:[[memory/Memory Management for Beginners]],
  [[execution/CPU Execution Provider]], [[observability/Tracing and Profiling]]
