---
title: Chunked Prefill
aliases:
  - 分块 Prefill
  - Chunked Prefill and Query Padding
  - Prefill 查询轴补齐
tags:
  - performance
  - prefill
  - attention
  - kv-cache
  - kernel-cache
status: maintained
created: 2026-08-19
updated: 2026-08-19
---

# 分块 Prefill,以及为什么查询轴必须补齐

> [!summary] 本文回答的问题
> Prefill 为什么可以按块喂进模型,而不需要"所有 KV 都参与"?分块换来了什么?
> 又留下了什么必须再补一刀才能收口的窟窿?

> [!important] 本文描述的是 `main` 上已实现的行为
> 表格里的数字都是在 `muse-glimmer-30b-int4`(ONNX INT4,native CUDA backend)上
> 实测的。与代码不一致的地方以代码为准。

Prefill 是"吃下 prompt"的那次前向,decode 是"吐出一个 token"的那次前向。
两者跑的是**同一张图**,只差一个数字:查询轴有几行。Prefill 一次交给模型整个
prompt(M 个 token 就是 M 行),decode 一次交一行。

就这一个数字,同时决定了一次请求的显存峰值,和运行时要编译多少个 kernel。

## 一、为什么 prompt 可以拆开喂

读一个 4000 token 的 prompt,最直白的做法是跑一次 `M = 4000` 的前向。分块 prefill
改成跑八次 `M = 512`,一片一片喂。**输出逐字节相同。**

这件事成立靠三条性质,缺一不可。

### 1. KV cache 是 append-only 的

位置 `i` 的 attention 要读所有 `≤ i` 位置的 key 和 value。而这些 key/value
只取决于**该位置的 token 和这一层的权重**,跟后面来什么完全无关。所以 cache 是
严格从左往右写的,一个位置写完之后再也不会变。

第 k 块跑的时候,`[0, k·W)` 已经被前面的块写进 cache 了。第 k 块算出自己这 W 个
位置的 key/value,追加在 `[k·W, (k+1)·W)`。前面的一个字节都不重访,一个字节都不重写。
第 k 块跑完后的 cache,和"一次性整块跑"所产生的 cache 的同长度前缀,**逐字节相同**。

这就是"为什么一个块不需要所有 KV 参与"的答案 —— 它**确实**在对目前为止所有的 KV
做 attention。它不需要的是**重新计算**那些 KV。前面的 KV 就躺在 cache 里被读,
这个块唯一的新工作量,是它自己那 W 行 query。

> [!tip] 一句话记法
> 分块省掉的不是"读",是"算"。

### 2. 因果掩码让切分不可见

Decoder-only 模型会掩掉位置 `i` 对任何 `> i` 位置的 attention。把整块跑和分块跑
并排看:对于第 k 块里位置为 `p` 的 query,

- 整块跑允许它看 `[0, p]`
- 分块跑允许它看 `[0, k·W) ∪ [k·W, p]`

**是同一个集合。** 分块跑还没算出来的那些位置,正是因果掩码本来就要抹掉的位置。

所以分块**不是近似**,不是拿质量换速度。算术完全一样,变的只是指令发射的顺序。

### 3. KV 张量的形状不动

如果 KV 张量每跑完一块就 resize 到当前真实序列长度,那每一块呈现给每个 attention
节点的形状都不一样,运行时会把每一个都当成新 kernel。

`GroupQueryAttention` 用**把容量和长度分开**绕过了这件事:`past_key` / `past_value`
在整个生成期间都按**物理容量**绑定,真实有效位置数由另一个输入 `seqlens_k` 携带。
这个事实在 `onnx-runtime-session` 里是写死的 ——
`crates/onnx-runtime-session/src/executor/geometry.rs` 里的
`kernel_input_uses_physical_capacity` 对 GQA 的 input 3 和 4 返回 `true`。

这条的实际后果值得说白:

> [!important] KV cache 变长,不改变任何 kernel 的 input shape
> 一次跑到 4000 token 的生成,和一次跑到 400 token 就停的生成,编译的 kernel 数量
> 一样多。

关于 KV cache 本身如何在不搬运、不换指针的前提下增长,见
[[memory/Virtual Memory for KV Cache]]。

## 二、分块是为了什么

Prefill 的 attention 需要 `O(M × total)` 的工作量,更要命的是需要 `O(M × total)`
的 scratch 去装 attention scores。`M = 4000` 时这块 scratch 大得吓人,而且是**瞬时**的
—— 一次前向分配,跑完就还。一个 30B 级别的模型,权重和 KV cache 都装得下,
完全可能**单单被 prefill 的这个尖峰**顶爆。

把 M 压到块宽 W,尖峰就压到 `O(W × total)`。代价是多几次 kernel launch、多几趟
过权重。块宽由模型在 inference metadata 里声明:

```yaml
runtime_configurable:
  chunked_prefill:
    chunk_size: 512
```

在本仓库里,`NativeDecodeSession::set_prefill_chunk_size` 读取它,
`decode_argmax`(`crates/onnx-genai-engine/src/native_decode/backend.rs`)在 prompt
长于一块时用 `token_ids.chunks(chunk)` 切开。

## 三、分块留下的窟窿

分块把 M 固定在 W —— **除了最后一块**,它拿到的是余数。

- 1137 token 的 prompt,W = 512,跑的是 `512, 512, 113`
- 1200 token 的 prompt,跑的是 `512, 512, 176`

余数是"prompt 长度对块宽取模",也就是说,**基本上每个请求一个新数字**。

这件事之所以要命,是因为 kernel cache 是按**节点 + 输入形状**做 key 的
(`crates/onnx-runtime-session/src/executor/kernel_cache.rs` 里的 `KernelKey`)。
一个没人跑过的查询宽度,会在图里**每一个节点**上 miss;对 30B decoder 来说,
就是每个请求重新编译约 890 个 kernel,永远如此。更糟的是,每个编译出来的 kernel
自己持有 device scratch,所以这个 cache 同时也是**资源治理器完全看不见的显存** ——
这就是 issue #1362 追踪的无界增长。

Cache 本身有自我约束:每个节点只保留最近的若干个 variant,其余淘汰。但**上限只有在
工作集装得进去的时候才有用**,而一个"每个请求一个新余数宽度"的工作集,永远装不进
任何上限。实测(五个不同长度的 prompt 各跑两遍):

| 第几次前向 | 查询行数 | 编译的 kernel 数 | 累计淘汰数 |
| --- | --- | --- | --- |
| 第一遍 | 104 | 1105 | 0 |
| 第一遍 | 146 | 888 | 0 |
| 第一遍 | 311 | 890 | 52 |
| **第二遍** | **146** | **888** | **5688** |
| **第二遍** | **311** | **888** | **7045** |
| **第二遍** | **37** | **888** | **9759** |

第二遍跑**完全相同的 prompt 长度**,重编译量和第一遍一样多,淘汰数还在线性上涨。
这个 cache 是在**抖动**,不是在缓存。

## 四、查询轴补齐

修法是把 GQA 已经在 KV 轴上用的那一招,搬到查询轴上。GQA 不把 cache resize 到精确
序列长度,而是按固定容量跑,把真实长度放在旁边带着。Prefill 完全可以照做:
**按取整后的宽度跑,把真实行数放在旁边带着。**

`prefill_query_width`(在 `crates/onnx-genai-engine/src/native_decode/cuda.rs`)
把一次前向的行数向上取整到块宽以下的三个等距档位 —— 块宽 512 时是
`{171, 342, 512}`。多出来的行用重复最后一个真实 token 填满,任何随请求送进来的
per-token 端口(`inputs_embeds`,以及其他形如 `[1, rows, …]` 的张量)按同样方式加宽。

补齐出来的行在算术上是纯浪费,但它们**不可能出错**:

- **读不到。** 因果掩码不让绝对位置 `p` 的真实行看到 `p` 之后,而每一个补齐行都排在
  所有真实行之后。
- **logits 被丢掉。** 结果在交给调用者之前就截回真实行数。
- **KV 被抹掉。** 前向跑完后 session rewind 到 `past_len + 真实行数`,这会把 attention
  掩码的尾巴清零、把 cache 的逻辑长度回滚。下一次前向直接覆盖写掉那些补齐条目。

> [!warning] 两种情况拒绝补齐,而不是冒险
> **带 recurrent / convolutional state 的 decoder 直接排除。** 那种状态既不受掩码
> 保护,也不由"逻辑长度"寻址,所以多推进一步就再也收不回来。
>
> **形状不是 `[1, rows, …]` 的送入端口**不能被认定为 per-token,整个补齐计划直接
> 作废,而不是替它凭空造几行出来。

另外还有一道运行期自检:如果一次补齐后的前向返回的 logits 行数**少于**交给它的
查询行数,说明这个 decoder 在内部把查询轴收缩了,它的行没法映射回输入位置 ——
此时对该 session 永久关闭补齐,并按原样重跑一次。

设 `ONNX_GENAI_PREFILL_QUERY_PADDING=0` 可以整体关掉。

## 五、为什么阶梯只有三档

把宽度集合变有界,只有在 kernel cache 的每节点上限**至少和这个集合一样大**时才有用。
那个上限是 4。

很容易想到的动作是把它调大。**那是个陷阱。** 保留下来的每个 variant 都持有 device
scratch,而这个上限是唯一给这些 scratch 封顶的东西。为了塞下一个八档阶梯而把上限
提到 10,结果是 30B decoder 一路爬到 72.8 GB,在 76.2 GB 的 mapped 上限前把
5.5k-token 的 prompt 直接跑挂 —— 而同一个 decoder 在上限为 4 时轻松服务了同一个
prompt,显存在 39–53 GB 之间振荡。

所以阶梯是**按已有的上限来定尺寸**,而不是反过来:三个 prefill 宽度,正好给
单 token decode 的那个形状留下一个槽。

| 配置 | 不同的 prefill 宽度数 | 第二遍重编译量 | 5.5k-token prompt |
| --- | --- | --- | --- |
| 不补齐,上限 4 | 无界 | 每次前向约 888 | ok,39–53 GB |
| 补齐(八档),上限 4 | 5 | 每次前向约 888 | — |
| 补齐(八档),上限 10 | 5 | 0 | **失败,72.8 GB** |
| 补齐(三档),上限 4 | 3 | **0** | **ok,平坦 ~50 GB** |

端到端延迟在 40 到 900 token 的各种 prompt 上都在噪声范围内没有变化,greedy 输出在
补齐开和关两种情况下逐字节相同。

> [!note] 这个改动的收益不是速度
> 收益是:一个长跑的 server 不再持续编译 kernel,也不再持续堆积它从来没记过账的
> kernel scratch。

## 六、档位数是怎么选的

三档是"在四 variant 上限里、还要给 decode 形状留一个槽"的前提下能放下的最大阶梯,
它带来的浪费就是这个约束的代价:一次前向最多多跑 170 行。

档位再少会更好缓存、但更浪费 —— 改用 2 的幂阶梯会把 311 行送到 512,这多出来的
65% 重复算术在中等长度 prompt 上是能测出来的(一个 200 词的 prompt 从 4.9 秒变成
7.0 秒)。

步长是**向上取整**的,这样最高一档正好落在块宽上,而不是差一点点、在上面又多留出
一个零散的第四种宽度。仓库里有一个测试专门守这条不变式,覆盖各种声明块宽。

## 相关阅读

- [[memory/Virtual Memory for KV Cache]] —— KV cache 如何在不搬运、不换指针的情况下增长
- [[memory/Memory Management for Beginners]] —— 分配、驻留、治理器的基础
- [[performance/Performance Engineering Playbook]] —— 怎么判断一个性能数字说的是运行时还是实验本身
- [[execution/CUDA Execution Provider]] —— 本文所有实测所在的执行路径
- [[metadata/Metadata Driven Runtime]] —— `chunked_prefill.chunk_size` 这类声明如何进入运行时
