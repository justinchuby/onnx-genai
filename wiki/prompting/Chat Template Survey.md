---
title: Chat Template Survey
aliases:
  - 对话模板横向调研
  - Chat Template Comparison
  - Tool Calling Formats
tags:
  - models
  - tokenizer
  - prompt
  - tools
  - survey
status: maintained
lang: zh-CN
created: 2026-08-19
updated: 2026-08-19
---

# Chat Template Survey

> [!summary] 本文回答的问题
> 各家模型的对话模板到底有多不一样?工具调用在模板层面是怎么落地的?一个转接层要
> 兼容它们,必须准备好应对哪些差异 —— 以及哪些差异**不会报错,只会悄悄丢数据**?

本文是 [[prompting/Chat Templates]] 的横向补充。那一篇讲机制,这一篇讲**分布**:
54 份真实模板,53 份能解析,49 份能完整渲染出一轮工具调用对话。

## 一、方法:渲染,而不是阅读

先说方法,因为本文所有结论的可信度都取决于它。

**读模板会读错。**Jinja 的分支很深,`{%- if builtin_tools is defined -%}` 这类条件
决定了同一份模板会渲染出完全不同的东西。本文的所有断言都不是从模板源码"看"出来的,
而是**真的把模板渲染出来**,再从输出里读。

统一的输入是同一段对话:一条 system、一个用户提问、一条带 `tool_calls` 的助手消息
(同时挂了 `reasoning_content` 与 `thinking`)、一条工具结果、一条助手回答、再一个
用户提问,`add_generation_prompt=True`。

为了让"到底渲染出来没有"可判定,输入里的每个字段都填**唯一哨兵串**
(`ZCITYZ`、`ZRESULTZ`、`ZREASONZ`……),然后在输出里查这些串在不在。

> [!warning] 这一步我做错过一次
> 第一版探测器用 `reasoning_content="R"`,再用"输出里有没有大写 R"来判断推理段是否
> 被保留。结果 `Reasoning: medium`、`RES` 这些无关文本全被算成命中,整列结论都是错的。
> **哨兵必须唯一到不可能偶然出现**,否则测的是巧合。

渲染环境用 Jinja2 的 `ImmutableSandboxedEnvironment`,补齐 HF 会注入的
`raise_exception`、`strftime_now`、`tojson`,并实现了 transformers 的
`{% generation %}` 标签(它用来标注训练时的 assistant mask,标准 Jinja2 没有,
不补的话 SmolLM3 这类模板连解析都过不去)。

复现用的最小骨架:

```python
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.exceptions import TemplateError
import json, datetime

env = ImmutableSandboxedEnvironment(
    trim_blocks=True, lstrip_blocks=True,
    extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(TemplateError(m))
env.globals["strftime_now"]    = lambda f: datetime.datetime.now().strftime(f)
env.filters["tojson"]          = lambda o, **k: json.dumps(o, ensure_ascii=False)

out = env.from_string(src).render(
    messages=msgs, tools=tools, add_generation_prompt=True,
    bos_token="<BOS>", eos_token="<EOS>")
```

### 样本与来源

模板取自 Hugging Face 各仓库的 `chat_template.jinja`,没有该文件的退回
`tokenizer_config.json` 的 `chat_template` 字段,抓取日期 2026-08-19。

> [!note] gated 模型走了镜像
> Llama、Gemma、Command-R 的官方仓库需要授权(HTTP 401)。Llama 与 Gemma 的模板取自
> `unsloth/*` 的再上传镜像,Command-R 两个镜像也是 401、**未纳入本文**。镜像与官方是否
> 逐字节一致本文没有验证,相关结论请按"来自镜像"打折。其余模型均为官方仓库。

## 二、轴一:一轮对话是怎么框起来的

54 份模板里,回合分隔的写法基本可以归到几族。下面每一行的标记都是从**渲染输出**里
摘的,不是从模板源码里摘的。

| 家族 | 代表 | 助手轮的框 |
|---|---|---|
| ChatML | Qwen 全系、Hermes、Olmo 3、Yi | `<\|im_start\|>assistant\n` … `<\|im_end\|>` |
| Llama 3.x | Llama 3.1 / 3.3 | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` … `<\|eot_id\|>` |
| Llama 4 | Llama 4 Scout | `<\|header_start\|>assistant<\|header_end\|>\n\n` … `<\|eot\|>` |
| Harmony 系 | gpt-oss、Muse Glimmer | `<\|start\|>assistant` … `<\|message\|>` … `<\|end\|>` / `<\|eot\|>` |
| 方括号 | Mistral 全系 | `[INST]` … `[/INST]` |
| Gemma | Gemma 2 / 3 | `<start_of_turn>model\n` … `<end_of_turn>` |
| 竖线角色 | GLM-4.5、EXAONE、Granite | `<\|assistant\|>`、`[\|assistant\|]`、`<\|start_of_role\|>assistant<\|end_of_role\|>` |
| 其它 | Kimi K2、MiniMax-M2、Nemotron | `<\|im_assistant\|>assistant<\|im_middle\|>`、`[e~[\n]~b]ai`、`<SPECIAL_11>Assistant` |

MiniMax-M2 的 `[e~[\n]~b]` 值得单独看一眼 —— 分隔符里带着字面的换行转义,这提醒一件事:
**不要假设分隔符是"看起来像标记"的字符串**,它只是训练时用过的一串 token 而已。

## 三、轴二:推理段 —— channel 与成对标签两派

### 一派:channel(Harmony 系)

gpt-oss 与 Muse Glimmer 用的是**同一族格式**。不是"类似",而是共享同一套骨架:

```text
<|start|>{sender} to={recipient} [<|channel|>{channel}] <|message|>{content}{stop}
```

两者渲染出来的工具调用逐字对比:

```text
gpt-oss      <|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Paris"}<|call|>
Muse Glimmer <|start|>assistant to=get_weather<|message|><atem:function_calls>…</atem:function_calls><|eot|>
```

`<|start|>`、`to=`、`<|message|>` 三件套完全一致。差别只有两处:

1. **gpt-oss 多一个正交的 `<|channel|>` 维度。**它同时有"收件人"(`to=`)和"频道"
   (`analysis` / `commentary` / `final`),两者独立;Muse Glimmer 把它们合并成一个
   `to=`,用 `to=self` 表达 gpt-oss 的 `analysis`,用 `to=user` 表达 `final`。
2. **终止符不同。**gpt-oss 用 `<|end|>` / `<|call|>` / `<|return|>` 三个;Muse Glimmer
   用 `<|eot|>` / `<|eom|>` 两个。

gpt-oss 甚至把频道清单写进 system 块,和 Muse Glimmer 写 `# Valid recipients:` 是同一个
手法:

```text
# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.
```

所以把它们看成同一种设计的两个变体是准确的。

### 另一派:成对标签

其余带推理能力的模型几乎都用**成对标签**,而不是 channel:

| 标签 | 模型 |
|---|---|
| `<think>` / `</think>` | Qwen3、QwQ、DeepSeek-R1、DeepSeek-V3.1、GLM-4.5、EXAONE 4.0、MiniMax-M2、Nemotron 系、Baichuan-M2、Ling、Skywork、MiniCPM 4.1 |
| `<seed:think>` | Seed-OSS |
| `[THINK]` / `[/THINK]` | Magistral |

### 两派的差别:标记在头上,还是在内容里

两派的差别不是审美。**channel 是结构化的,成对标签是文本内的。**用标签时,运行时要在
生成的文本流里做字符串匹配才知道推理段在哪结束;用 channel 时,收件人写在消息头上,
解析器读到头就知道了,不需要等内容。

### 一个更要紧的差别:谁负责丢掉历史推理段

推理段不应该回填进后续轮次(见 [[prompting/Chat Templates]])。但**这件事由谁来做,
各家不一样**,而这直接决定了转接层要不要自己动手。

实测:同一段带 `reasoning_content` 与 `thinking` 的历史,渲染后哨兵串还在不在 ——
49 份可渲染的模板里,只有 **2 份**保留了它:`ByteDance-Seed/Seed-OSS-36B-Instruct` 和
`meta-models/Muse-Glimmer-30B`。

也就是说:

- **绝大多数模板自己就把历史推理段丢了。**gpt-oss 的模板里甚至写着注释
  *"CoT is dropped during all previous turns, so we never render it for inference"*。
  你传了也白传,不会有副作用。
- **Seed-OSS 和 Muse Glimmer 会忠实渲染传入的推理段。**对这两个模型,"不回填历史推理"
  是**调用方的责任**;照着"反正模板会丢"的假设写代码,在这两个模型上就会把历史思考
  全部塞回 context。

> [!important] 结论
> "推理段不回填"这条规则不能依赖模板来兑现。要么调用方自己在构造 `messages` 时就不带
> 历史 `reasoning_content`,要么就得逐模型确认。前者永远是对的,后者要维护一张表。

## 四、轴三:工具调用

这是差异最大的一轴。它其实是**三段互相独立的机制**,一个模型可以只实现其中一部分 ——
这正是下一节那些静默故障的来源。

### 第 1 段:工具定义注入到哪里

| 位置 | 模型 |
|---|---|
| system 块内 | Qwen 全系、GLM-4.5、Kimi K2、Granite、Muse Glimmer、Seed-OSS |
| 独立的 developer 轮 | gpt-oss |
| **user 轮** | Llama 3.1 / 3.3 |
| 独立的 `available_tools` 轮 | Granite 3.3 |
| 独占的 `[AVAILABLE_TOOLS]` 段 | Mistral 全系 |
| 自建 system 块(覆盖在调用方的 system 之前) | Hermes 3 |
| **完全不注入** | DeepSeek-R1、DeepSeek-V3.1 |

两个反直觉的:

**Llama 3.1 把工具定义写进 user 轮**,不是 system:

```text
<|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call ...
{"type": "function", "function": {"name": "get_weather", ...}}

Weather in Paris?<|eot_id|>
```

**DeepSeek-R1 / V3.1 直接忽略 `tools` 参数。**实测:工具描述的哨兵串完全没出现在输出里,
但它们**能**渲染 `tool_calls` 和工具结果。也就是说这两个模型的工具定义**必须由调用方
自己写进 system prompt**,传 `tools=[...]` 什么也不会发生,而且不报错。

描述的格式也不统一。gpt-oss 把 JSON Schema 转成了 **TypeScript 声明**:

```text
namespace functions {
// Get weather.
type get_weather = (_: {
// City
city: string,
}) => any;
} // namespace functions
```

Muse Glimmer 则写成大段自然语言 + XML 用法示例 + schema 清单。Hermes 3 会把 schema
连同一段 pydantic 模型说明一起塞进去,并且**另起一个 system 轮**放调用方的 system —— 
所以渲染结果里出现了**两个连续的 system 块**。

### 第 2 段:助手发起调用的语法

全部摘自渲染输出,逐字:

| 模型 | 助手发起调用的样子 |
|---|---|
| Qwen3 / Hermes / Granite 4 / EXAONE / Ling | `<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>` |
| Llama 3.1 / 3.3 | `{"name": "get_weather", "parameters": {"city": "Paris"}}` (裸 JSON,注意键是 `parameters`) |
| **Llama 4** | `[get_weather(city="Paris")]` (Python 调用语法) |
| **Olmo 3** | `<function_calls>get_weather(city="Paris")</function_calls>` |
| Mistral v0.3 | `[TOOL_CALLS] [{"name": …, "arguments": {…}, "id": "abcdefghi"}]` |
| Mistral Small 3.2 | `[TOOL_CALLS]get_weather[CALL_ID]call_abc12[ARGS]{"city": "Paris"}` |
| Magistral | `[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}` |
| gpt-oss | `<\|start\|>assistant to=functions.get_weather<\|channel\|>commentary json<\|message\|>{"city": "Paris"}<\|call\|>` |
| Muse Glimmer | `<atem:invoke name="get_weather"><atem:parameter name="city">Paris</atem:parameter></atem:invoke>` |
| MiniMax-M2 | `<minimax:tool_call><invoke name="get_weather"><parameter name="city">Paris</parameter></invoke></minimax:tool_call>` |
| step3 | `<steptml:invoke name="get_weather"><steptml:parameter name="city">Paris</steptml:parameter></steptml:invoke>` |
| Seed-OSS | `<seed:tool_call><function=get_weather><parameter=city>Paris</parameter></function></seed:tool_call>` |
| GLM-4.5 | `<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>` |
| Kimi K2 | `<\|tool_calls_section_begin\|><\|tool_call_begin\|>call_abc12<\|tool_call_argument_begin\|>{"city": "Paris"}<\|tool_call_end\|>` |
| DeepSeek-V3.1 | `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Paris"}<｜tool▁call▁end｜>` |
| Nemotron Nano | `<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</TOOLCALL>` |
| Falcon3 | `<tool_call>\n[{"id": …, "type": "function", "function": {…}}]\n</tool_call>` |

几个能直接省掉排查时间的点:

- **`<tool_call>` + JSON 是事实上的多数派**,Qwen 的写法被大量沿用。想只支持一种解析器
  时,先支持它。
- **Llama 3.1 用的键是 `parameters` 而不是 `arguments`。**跟 OpenAI wire format 不一样。
- **XML 式调用正在变多。**Muse Glimmer(`atem:`)、MiniMax(`minimax:`)、step3
  (`steptml:`)、Seed-OSS 都是同一个思路,只是换了命名空间前缀。它们的共同动机是
  避免让模型输出严格合法的 JSON —— Muse Glimmer 的模板里直接写了
  *"The output is not expected to be valid XML and is parsed with regular expressions."*
- **Llama 4 和 Olmo 3 让模型写函数调用代码**,不写 JSON。解析这两种要写表达式解析,
  不能用 `json.loads`。

> [!warning] `<|python_tag|>` 是个常见误解
> 网上(以及本 wiki 之前的版本)常说 Llama 3.1 用
> `<|python_tag|>{...}<|eom_id|>` 发起工具调用。**默认路径不是这样。**
> 读模板可以看到,`<|python_tag|>` 只在 `builtin_tools is defined`
> **且**被调函数在 `builtin_tools` 里(`brave_search`、`wolfram_alpha` 这类内置工具)
> 时才出现;`<|eom_id|>` 同样只在 `builtin_tools is defined` 时才用。
> 只传普通的 `tools=[...]`,渲染出来是**裸 JSON + `<|eot_id|>`**,上表就是实测结果。

### 第 3 段:工具结果怎么回填

角色名比调用语法还乱:

| 结果所用的角色 | 模型 |
|---|---|
| `tool` | Hermes 3、Muse Glimmer(`<\|start\|>tool get_weather`) |
| **`user`** | Qwen 全系、Granite 4、Nemotron |
| `ipython` | Llama 3.1 / 3.3 / 4 |
| `observation` | GLM-4.5(`<\|observation\|>`) |
| `OBSERVATION` | Ling |
| `environment` | Olmo 3 |
| `[\|tool\|]` | EXAONE 4.0 |
| `<\|im_system\|>{函数名}` | Kimi K2 |
| `functions.{函数名} to=assistant` | gpt-oss |
| 专用段落,无角色 | Mistral(`[TOOL_RESULTS]`)、DeepSeek(`<｜tool▁output▁begin｜>`) |
| **`assistant`** | Falcon3 |

**Qwen 把工具结果渲染成 `user` 轮**是最容易踩的一个 —— 模板里是
`<|im_start|>user\n<tool_response>\n…\n</tool_response>`。连带的两个细节:

1. 连续多条工具结果会被**合并进同一个 user 轮**(模板里判断前一条是不是也是 `tool`)。
2. Qwen3 为了找到"最后一条真正的用户消息",要靠**检测 user 消息是不是被
   `<tool_response>` 包着**来把工具结果排除掉:
   ```jinja
   {%- if ns.multi_step_tool and message.role == "user" and message.content is string
        and not(message.content.startswith('<tool_response>')
        and message.content.endswith('</tool_response>')) %}
   ```
   也就是说,**如果用户自己发了一条以 `<tool_response>` 开头结尾的消息,Qwen3 会把它
   误判成工具结果。**这是模板层面真实存在的歧义。

Falcon3 把工具结果挂在 `<|assistant|>` 头下面(模板第 24 行),等于让助手"自己说出"了
工具的输出。这个我倾向于认为是模板缺陷,而不是设计。

## 五、输入约定:同样是 OpenAI 形状,并不通用

调用方给的 `messages` / `tools` 结构看起来是统一的 OpenAI 形状,但模板对它的期待不一致。
下面每一格都是**分别渲染验证过的**(基线配置跑通后,单独翻转一个轴再跑一次),不是从
"第一个能跑通的组合"反推的 —— 那样只能证明某种写法可行,不能证明另一种会失败。

| 约定 | 结论 |
|---|---|
| `arguments` 传 **dict** | 47/49 接受。**DeepSeek-R1、DeepSeek-V3.1 报错**(它们要 JSON 字符串) |
| `arguments` 传 **JSON 字符串** | **GLM-4.5、MiniMax-M2、Seed-OSS、Olmo 3、step3、Muse Glimmer 报错** |
| `tool_call_id` 非 9 字符 | **Mistral v0.3、Mistral-Nemo 报错**:*"Tool call IDs should be alphanumeric strings with length 9!"* |
| `tools` 用扁平结构(不套 `function`) | gpt-oss、MiniMax-M2、Mistral v0.3 报错;**Apertus 反过来只接受扁平结构** |

这张表最实用的一行是第一、二行合起来看:**不存在一种 `arguments` 写法能通吃。**
dict 会被 DeepSeek 拒,字符串会被另外六个拒。转接层必须按模型转换。

Qwen3 是唯一在模板里显式处理了两种写法的:

```jinja
{%- if tool_call.arguments is string %}
    {{- tool_call.arguments }}
{%- else %}
    {{- tool_call.arguments | tojson }}
{%- endif %}
```

实测两种输入渲染出来逐字节相同。这是防御性写法的好例子:**它把一个转接层的负担
吸收进了模板**。

Mistral 的 9 字符 ID 是最著名的一个坑。OpenAI 的 `call_abc123...` 形式的 ID 会直接
触发异常。注意这条**只在老模板里**:Mistral-Small-3.2、Devstral、Magistral 已经取消了
这个检查。

## 六、静默失败:比报错危险得多

上面那些至少会抛异常。真正危险的是**渲染成功、但内容没了**。

### 6.1 渲染工具定义,却丢掉工具调用

这三份模板会把 `tools` 渲染进 prompt,但**整份模板里没有任何一处引用
`message.tool_calls`**(实测 grep 命中数为 0):

- `ibm-granite/granite-3.3-8b-instruct`
- `HuggingFaceTB/SmolLM3-3B`
- `zai-org/GLM-4-9B-0414`

后果:一段"助手调用了工具 → 工具返回 → 助手回答"的历史,渲染出来是

```text
<|start_of_role|>assistant<|end_of_role|><|end_of_text|>
```

**助手轮变成空的**,工具调用凭空消失。模型看到的历史是"它什么都没说,然后工具结果就
出现了"。不报错,不警告。

它们的设计意图是让模型把调用**写在 `content` 里**(Granite 3.3 的 system 提示写着
*"respond only with `<|tool_call|>` followed by a JSON list of tools used"*),所以
调用方必须把工具调用作为文本放进 `content`,而不是放进结构化的 `tool_calls` 字段。

### 6.2 忽略 `tools` 参数

`deepseek-ai/DeepSeek-R1` 和 `deepseek-ai/DeepSeek-V3.1`:传 `tools=[...]`,渲染输出里
工具描述的哨兵串**完全不存在**。工具定义必须自己写进 system prompt。

### 6.3 完全没有工具支持

12 份模板既不渲染定义也不渲染调用:Yi-1.5、ChatGLM3、OLMo-2、ERNIE-4.5、
DeepSeek-V2-Lite、InternLM3、Phi-3.5-mini、Phi-4、Phi-4-mini、MiniCPM-4.1、
dots.llm1、Hunyuan-A13B。

**Gemma 2 / 3 更严格**:模板里根本没有 `tool` 角色,而且强制 user/assistant 交替,
一旦历史里出现 `tool` 就直接抛
*"Conversation roles must alternate user/assistant/user/assistant/..."*。
Gemma 2 连 system 角色都不支持(*"System role not supported"*)。用 Gemma 做工具调用,
整套协议得自己在 user 轮里搭。

> [!important] 检查清单里应该加一条
> 接入一个新模型时,**渲染一段带工具调用的历史,然后确认调用真的出现在输出里**。
> 这一步只要三行代码,能挡掉本节所有情况。只测"渲染没报错"是不够的。

## 七、模板自己带的 bug

模板是随模型发布的数据文件,和代码一样会有缺陷,而且很少有人去测。

**`stepfun-ai/step3` 的分隔符前后不一致。**模板里教给模型的写法是
`<｜tool_call_begin｜>`(出现 2 次,都在给模型看的说明文字里),但**真正渲染历史时**
用的是 `<｜tool_call_begin>｜>` —— `>` 和 `｜` 写反了。也就是说,模型被教了一种写法,
而它自己历史里的记录是另一种。

**`tiiuae/Falcon3` 把工具结果放在助手头下面**,见上一节。

这两个都不是我推测的,是渲染输出和模板源码逐字对照出来的。它们的意义不在于这两个模型
本身,而在于:**模板不能假定是对的。**接入时把渲染结果打出来看一眼,比读文档可靠。

## 八、对本仓库的影响

对照 `crates/onnx-genai-ort/src/chat_template.rs` 与 `crates/onnx-genai/src/reasoning.rs`:

1. **`pycompat` 是必需的,不是优化。**本次 49 份成功渲染里,大量模板调用了 Python 的
   字符串方法。这印证了现有实现挂 `minijinja_contrib::pycompat` 的决定。
2. **`raise_exception` 必须实现。**样本里 Mistral-Small-3.2 有 8 处、Apertus 有 11 处、
   gpt-oss 与 Mistral v0.3 各 4 处。不实现它 = 静默忽略模型声明的硬约束。
3. **`{% generation %}` 需要容错。**SmolLM3 用了这个 transformers 扩展标签。纯 Jinja
   实现会在**解析阶段**就失败,连降级机会都没有。
4. **`arguments` 需要按模型双向转换。**现状是 dict 和字符串都有模型强制要求,没有通用解。
   这比 [[prompting/Chat Templates]] 里"HF 要 dict、OpenAI 要字符串"的说法更复杂:
   HF 阵营内部就不统一。
5. **`ChatRole::Other(String)` 的设计被证实是必要的。**实测出现过的结果角色至少有
   `ipython`、`observation`、`OBSERVATION`、`environment` 四种非标准值,还有 Qwen 把
   结果塞进 `user`、Kimi 塞进 `im_system` 这类完全不走角色枚举的写法。
6. **推理段的丢弃不能依赖模板。**见第三节:只有 2/49 会回放,但正因为存在这 2 个,
   调用方侧的丢弃逻辑不能省。
7. **`reasoning.rs` 目前只识别成对分隔符。**Harmony 系(gpt-oss、Muse Glimmer)用的是
   消息头上的收件人/频道,需要在解析层识别 `<|start|>` 之后、`<|message|>` 之前的那段。
   这仍是一个未实现的差距。

## 九、给转接层的最小结论

1. **不存在通用的 `arguments` 形状。**按模型转 dict / JSON 字符串。
2. **不存在通用的工具结果角色。**至少要能表达 `tool` 以外的自定义角色。
3. **不要假设模板会用传入的 `tools` 参数。**DeepSeek 系不会。
4. **不要假设模板会渲染传入的 `tool_calls`。**Granite 3.3、SmolLM3、GLM-4-9B 不会,且不报错。
5. **接入验收 = 渲染一遍并检查哨兵串在不在**,不是"没抛异常"。
6. **`tool_call_id` 尽量用 9 位字母数字**,可以避开老 Mistral 的检查,对别家无害。
7. **推理段在调用方就丢掉**,别指望模板。

## 参考与出处

- 全部模板取自 Hugging Face 对应仓库的 `chat_template.jinja` 或
  `tokenizer_config.json`,抓取日期 2026-08-19。Llama / Gemma 取自 `unsloth/*` 镜像
  (官方仓库 gated)。
- 本文所有语法片段均为**渲染输出**的逐字摘录,渲染骨架见第一节。
- 相关实现:`crates/onnx-genai-ort/src/chat_template.rs`、
  `crates/onnx-genai/src/reasoning.rs`

## 相关笔记

- [[prompting/Chat Templates]] —— 机制篇:模板如何工作、多模态与工具调用的原理
- [[architecture/Inference Request Lifecycle]] —— 渲染之后请求的完整路径
- [[memory/Virtual Memory for KV Cache]] —— 渲染出的 token 在显存里如何安置
