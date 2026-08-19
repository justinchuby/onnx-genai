---
title: Using this Wiki
aliases:
  - Wiki Conventions
  - Obsidian Setup
tags:
  - wiki
  - obsidian
  - contributing
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-19
---

# Using this Wiki

> [!summary] 回答的问题
> 贡献者应如何在 Obsidian 中打开、链接、标注日期并维护这个仓库 wiki?

## 打开 vault

把仓库的 `wiki/` 目录作为一个 Obsidian vault 打开。这些笔记使用标准 Markdown、
YAML properties、wikilink、callout 与 Mermaid 图。阅读它们无需任何社区插件。

## 命名

- 使用英文文件名与英文 `title` property,以获得稳定的跨语言链接。
- 正文使用简体中文;文件名、frontmatter 的 `title` 以及页面的一级标题(H1)保持
  英文。
- 使用描述性的名词短语,而不是 issue 编号或临时的项目阶段。
- 按持久的领域组织:`start/`、`architecture/`、`execution/`、`memory/`、
  `meta/`,以及未来的同级领域。

## 必需的 property

```yaml
---
title: Stable English Title
aliases:
  - Optional alternate title
tags:
  - domain
status: maintained
lang: zh-CN
created: 2026-08-17
updated: 2026-08-17
---
```

`lang` 是必需字段。Quartz 会直接把它的值用作发布页面的 `<html lang>` 属性(参见
`site/quartz/quartz/components/renderPage.tsx` 中对 `frontmatter?.lang` 的使用),
因此它影响的是真实的可访问性与搜索引擎行为。中文笔记写 `lang: zh-CN`。

建议的 status 取值:

| Status | 含义 |
|---|---|
| `maintained` | 力求跟踪当前的理解 |
| `proposed` | 解释一个尚未完全实现的目标设计 |
| `historical` | 保留的背景,而非当前指南 |
| `draft` | 尚不完整,不宜依赖 |

## Obsidian 能显示创建与修改时间吗?

可以,但有两种不同的含义:

### 受版本控制的 property

`created` 与 `updated` 会出现在 Obsidian 的 Properties 视图中,并存储在 Git 里。
它们是这个 wiki 的持久日期。

- 新增笔记时设置 `created`。
- 当实质性地改变了笔记含义时,更改 `updated`。
- 对仅涉及空白或仅涉及链接的维护,不要更改 `updated`,除非本仓库采用了不同的
  约定。

Obsidian Core 不会在笔记每次变动时可靠地自动更新自定义的 `updated` property。
Linter/Templater 一类的插件或仓库自动化可以做到这一点,但贡献者不应为了产出合规
的笔记而必须依赖插件。

### 文件系统时间戳

Obsidian 与插件可以检视本地文件的创建/修改时间戳。它们在本地有用,但作为仓库历史
并不可靠:

- clone 会创建新的本地文件;
- checkout/rebase 可能改变 mtime;
- 不同的文件系统对元数据的保留方式不同;
- Git 不对文件系统的创建时间做版本控制。

要获得精确的变更来源,请使用 Git 历史:

```bash
git log --follow -- "wiki/path/Note.md"
```

> [!important]
> frontmatter 中的日期描述的是笔记级别的编辑史。Git commit 仍然是"谁在何时改动了
> 哪些行"的权威记录。

## 链接

对 wiki 内的概念,优先使用 wikilink:

```markdown
[[architecture/Crate Architecture]]
[[execution/Execution Backends|backend overview]]
```

对 `docs/` 或 `crates/` 下的源文件,使用普通的相对 Markdown 链接,因为 GitHub 能
正确渲染它们,且它们代表正式来源:

```markdown
[Memory Architecture](../../docs/memory/MEMORY_ARCHITECTURE.md)
```

## 预览并发布静态站点

发布后的站点直接读取 `wiki/`;没有第二份需要维护的笔记副本。在仓库根目录:

```bash
cd site/quartz
npm ci
npm run wiki:serve
```

打开 <http://localhost:8080/onnx-genai/>。在发起 pull request 之前,在同一目录运行
`npm run wiki:build`。该命令会校验 wikilink、构建生产站点,并检查生成的链接与资源
是否都保持在 `/onnx-genai/` 这个项目路径之下。

站点设置位于 `site/quartz/quartz.config.yaml`;依赖版本记录在相邻的 npm 与 Quartz
lockfile 中。`Wiki Pages` 工作流会为相关的 pull request 与 push 进行构建,但只有推
送到 `main` 才能部署到 GitHub Pages。仓库维护者必须在 Pages 来源中选择
**GitHub Actions**。

## 人类阅读预算

每篇笔记都应回答一个主要问题,通常大约需要 5–10 分钟读完。这是一种导航辅助,而不
是硬性的篇幅上限。当初学者需要在一条连续的阅读路径里获得定义与推理时,保留较长的
教程体例。

优先采用一张由链接笔记组成的小地图,而不是一整章试图保留每个实现细节的内容;但也
不要让读者仅仅为了读懂本文就去追逐链接。一篇 wiki 笔记应当对其目标读者自成一体。
形式化规范与源代码支撑的是验证与更深入的实现工作;它们不是理解本文的前置条件。

## 避免重复的事实

不要把大量会变动的状态表或详尽的规范性契约拷贝进 wiki。但要包含足够的定义、示例
与约束,让人无需打开另一个文件就能理解这个主题。链接权威来源,使论断可被核验,并
让维护者能够找到实现细节。

## 笔记模板

```markdown
---
title: Note Title
aliases: []
tags:
  - domain
status: draft
lang: zh-CN
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Note Title

> [!summary] 本文回答的问题
> 一句话说明读者能从这篇笔记里得到什么。

## 说明

...

## 权威来源

- [来源](../../docs/path.md)

## 相关笔记

- [[start/Repository Map]]
```
