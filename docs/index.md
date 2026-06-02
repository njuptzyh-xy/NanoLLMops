# NanoLLMOps 文档索引

更新时间：2026-06-02

## 开发前阅读顺序

Codex 每次开始开发前，必须按以下顺序读取：

1. `AGENTS.md`
2. `docs/index.md`
3. `docs/product/PRD.md`
4. `docs/product/architecture.md`
5. `docs/planning/roadmap.md`
6. `docs/planning/tasks.md`
7. 当前任务的 `docs/tasks/<任务编号>/spec.md`
8. 当前任务关联的 ADR、阶段总结和运行手册

## 文档目录

```text
docs/
├── index.md
├── product/
│   ├── PRD.md
│   └── architecture.md
├── planning/
│   ├── roadmap.md
│   └── tasks.md
├── tasks/
│   └── <任务编号>/
│       ├── spec.md
│       └── logs/
│           └── YYYY-MM-DD-HHmm.md
├── phases/
├── runbooks/
├── adr/
└── archive/
```

## 文档职责

| 文档类型 | 路径 | 回答的问题 |
| --- | --- | --- |
| 产品需求 | `docs/product/PRD.md` | 为什么做、为谁做、范围是什么 |
| 技术架构 | `docs/product/architecture.md` | 系统如何拆分、模块如何协作 |
| 路线图 | `docs/planning/roadmap.md` | 当前处于哪个阶段 |
| 任务清单 | `docs/planning/tasks.md` | 下一项任务是什么、状态如何 |
| 任务规格 | `docs/tasks/<任务编号>/spec.md` | 单个任务做什么、如何验收 |
| 开发日志 | `docs/tasks/<任务编号>/logs/*.md` | 某次开发实际执行了什么 |
| 阶段总结 | `docs/phases/*.md` | 某阶段整体完成了什么 |
| 运行手册 | `docs/runbooks/*.md` | 当前系统如何运行和验证 |
| ADR | `docs/adr/*.md` | 关键技术决策为什么这样做 |
| 历史归档 | `docs/archive/*.md` | 保存旧文档，不作为当前入口 |

## 当前入口

- 当前路线图：`docs/planning/roadmap.md`
- 当前任务清单：`docs/planning/tasks.md`
- 当前待领取任务：`P2-001`
- 当前任务规格：`docs/tasks/P2-001/spec.md`
- 当前运行手册：`docs/runbooks/mvp-e2e.md`
- 当前阶段总结：`docs/phases/PHASE-01-02-转换产物复用与runtime接入.md`

## 编号规则

| 类型 | 格式 | 示例 |
| --- | --- | --- |
| 阶段 | `PHASE-XX` | `PHASE-02` |
| 开发任务 | `P<阶段>-<序号>` | `P2-001` |
| 文档治理任务 | `DOC-<序号>` | `DOC-002` |
| 架构决策 | `ADR-<序号>` | `ADR-001` |

日期只属于日志文件，不属于任务身份。
