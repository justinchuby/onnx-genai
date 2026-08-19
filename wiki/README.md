---
title: Wiki
aliases:
  - Knowledge Base
tags:
  - wiki
  - index
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# onnx-genai Wiki

本目录是一个兼容 Obsidian 的知识库,收录解释性笔记、学习路径,以及实现概念之间的
关联。它**不替代** `docs/` 下的规范、实测证据与已接受的设计文档。

发布后的读者请从 [[index|onnx-genai Knowledge Base]] 开始。

> [!important] 来源优先级
> 当 wiki 笔记与正式文档或代码不一致时,按以下顺序采信:
> 1. 当前代码与可复现的实测数据
> 2. `docs/` 下的权威文档
> 3. 已接受的设计决策
> 4. 解释性的 wiki 笔记

## 内容地图

- **从这里开始:** [[start/Repository Map]]
- **架构:** [[architecture/Crate Architecture]]
- **运行时流程:** [[architecture/Inference Request Lifecycle]]
- **执行:** [[execution/Execution Backends]]
- **EP 契约:** [[execution/Execution Provider Contract]]
- **CPU EP:** [[execution/CPU Execution Provider]]
- **CUDA EP:** [[execution/CUDA Execution Provider]]
- **插件 EP:** [[execution/Plugin Execution Providers]]
- **内存:** [[memory/Memory Management for Beginners]]
- **KV cache 虚拟内存:** [[memory/Virtual Memory for KV Cache]]
- **MoE 路由倾斜:** [[memory/MoE Router Skew and Always-On Experts]]
- **对话模板:** [[prompting/Chat Templates]]
- **追踪:** [[observability/Tracing and Profiling]]
- **性能工程:** [[performance/Performance Engineering Playbook]]
- **分块 prefill:** [[performance/Chunked Prefill]]
- **API 设计:** [[api/API Design Principles]]
- **契约:** [[contracts/Runtime Contracts]]
- **形式化验证:** [[contracts/Formal Verification with TLA+]]
- **测试与验证:** [[development/Testing and Verification]]
- **元数据:** [[metadata/Metadata Driven Runtime]]
- **模型包:** [[metadata/Model Packages and Variants]]
- **文档:** [[start/Documentation Guide]]
- **Wiki 维护:** [[meta/Using this Wiki]]

## 笔记写作规范

每篇笔记都应当:

1. 使用**英文文件名**和**英文 `title`**,使链接在跨语言时保持稳定。
2. 包含 YAML frontmatter,字段为 `title`、`aliases`、`tags`、`status`、`lang`、
   `created`、`updated`。
3. 开篇先用一句话说明这篇笔记回答的是什么问题。
4. 只回答一个主要问题,通常保持在 5–10 分钟可读完的篇幅。
5. 当缩短会迫使初学者跑到其他文件里去补前置知识时,保留较长的教程体例。
6. 用 `[[wikilinks]]` 链接到其他笔记,而不是在多篇之间重复同一段解释。
7. 用 Obsidian callout 标注不变量、警告、示例与背景。
8. 对目标读者做到自包含。指向 `docs/` 和代码的链接是**证据和实现细节**,不是必须
   先读完的功课。
9. 明确标注"提议中"的行为,**绝不**把目标设计写成已实现的样子。

## 语言

> [!important] Wiki 正文用中文写,标题保留英文
> 本 wiki 的**正文使用简体中文**。以下内容**保持英文**:
>
> - **文件名与目录路径** —— 例如 `execution/CUDA Execution Provider.md`
> - **frontmatter 的 `title` 字段**
> - **页面的一级标题(H1)** —— 与 `title` 保持一致
> - **`tags`** —— 标签始终用英文
> - **`[[wikilinks]]` 的链接目标** —— 需要中文显示文本时用 `[[目标|显示文本]]`,
>   绝不改动竖线左边的目标本身
> - **代码、标识符、crate 名、文件路径、环境变量、函数名、命令行**
> - **Obsidian callout 的类型关键字** —— 如 `> [!important]`,这是语法
>
> 章节标题(H2 及以下)、正文、表格内容、callout 的标题文字都应译为中文。
>
> `aliases` 可以同时包含中英文条目 —— 它的作用是让人用任一语言都能搜到这篇笔记,
> 因此中文别名是鼓励的,但**不要**删除既有的英文别名,那会破坏已有链接。
>
> 技术术语首次出现时保留英文原词并在括号中给出中文,例如
> "execution provider(执行提供者,EP)";此后可只用英文缩写。当英文术语本身就是
> 业界通用称呼时(如 kernel、arena、allocator),直接沿用英文,不必强译。

`lang` 字段会被 Quartz 直接用作发布页面的 `<html lang>` 属性
(`renderPage.tsx`),因此它影响的是真实的可访问性与搜索引擎行为,不只是元数据。
中文笔记写 `lang: zh-CN`。

> [!note] 创建与修改日期
> Obsidian 可以在 Properties 视图里显示受版本控制的 `created` 与 `updated` 属性。
> Obsidian 同样知道本地文件系统的创建与修改时间,但那些时间在 clone、checkout 和
> rebase 之后并不可靠。Obsidian 核心功能**不会**在每次编辑时自动维护自定义的
> `updated` frontmatter;请随笔记一起更新它,或配置自动化/插件来处理。参见
> [[meta/Using this Wiki]]。
