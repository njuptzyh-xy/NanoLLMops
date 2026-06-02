# NanoLLMOps Development Agents Guide

本文件定义 NanoLLMOps 项目的默认开发范式。所有开发、维护、修复、重构和收尾工作都必须遵守。

## 1. 总目标

开发过程必须可追踪、可解释、可回溯。每项代码或文档改动必须归属于唯一任务编号，并具备规格、日志、验证、提交和推送记录。

## 2. 文档唯一入口

每次开始开发前，必须按以下顺序读取：

1. `AGENTS.md`
2. `docs/index.md`
3. `docs/product/PRD.md`
4. `docs/product/architecture.md`
5. `docs/planning/roadmap.md`
6. `docs/planning/tasks.md`
7. 当前任务的 `docs/tasks/<任务编号>/spec.md`
8. 当前任务关联的 ADR、阶段总结和运行手册

如果任务清单中的当前任务不明确，先整理任务清单，不直接修改业务代码。

## 3. 文档分层

| 文档类型 | 路径 | 职责 |
| --- | --- | --- |
| 产品需求 | `docs/product/PRD.md` | 定义业务目标、用户、范围和非目标 |
| 技术架构 | `docs/product/architecture.md` | 定义模块边界和技术方向 |
| 路线图 | `docs/planning/roadmap.md` | 定义阶段目标和推进顺序 |
| 任务清单 | `docs/planning/tasks.md` | 维护任务编号和状态，是唯一状态源 |
| 任务规格 | `docs/tasks/<任务编号>/spec.md` | 定义单个任务范围和验收标准 |
| 开发日志 | `docs/tasks/<任务编号>/logs/*.md` | 记录真实开发操作和结果 |
| 阶段总结 | `docs/phases/*.md` | 汇总阶段完成项、边界和后续重点 |
| 运行手册 | `docs/runbooks/*.md` | 保存可执行命令和验证路径 |
| ADR | `docs/adr/*.md` | 记录关键技术决策及原因 |
| 历史归档 | `docs/archive/*.md` | 保存旧记录，不作为当前入口 |

不同类型文档不允许混写。任务清单不追加开发流水，任务日志不维护另一套任务状态。

## 4. 编号规则

编号必须稳定，不因日期或文件移动而变化。

| 类型 | 格式 | 示例 |
| --- | --- | --- |
| 阶段 | `PHASE-XX` | `PHASE-02` |
| 开发任务 | `P<阶段>-<序号>` | `P2-001` |
| 文档治理任务 | `DOC-<序号>` | `DOC-002` |
| 架构决策 | `ADR-<序号>` | `ADR-001` |

日志文件使用时间戳：

```text
docs/tasks/<任务编号>/logs/YYYY-MM-DD-HHmm.md
```

同一任务跨多次开发时新增日志文件，不覆盖历史日志。

## 5. 任务状态

`docs/planning/tasks.md` 是唯一任务状态源。只允许使用：

- `TODO`
- `IN_PROGRESS`
- `BLOCKED`
- `DONE`
- `CANCELLED`

开始开发前将任务更新为 `IN_PROGRESS`。满足验收标准后才能更新为 `DONE`。

## 6. 单任务开发流程

每次只围绕一个任务编号开发：

1. 从 `docs/planning/tasks.md` 领取任务。
2. 阅读或创建 `docs/tasks/<任务编号>/spec.md`。
3. 将任务状态更新为 `IN_PROGRESS`。
4. 实施代码或文档改动。
5. 执行与改动直接相关的验证。
6. 写入 `docs/tasks/<任务编号>/logs/YYYY-MM-DD-HHmm.md`。
7. 满足验收标准后，将任务状态更新为 `DONE`。
8. 必要时更新 runbook、ADR、architecture 或 phase 总结。
9. 检查 `git diff` 和 `git status`。
10. 创建 Git 提交。
11. 推送当前分支到远程仓库。
12. 汇报结果。

## 7. 任务日志要求

每个日志至少覆盖：

- 任务编号和任务名称
- 本次目标
- 背景或目的
- 处理逻辑
- 实际改动文件
- 验证命令
- 验证结果
- 当前结论
- 未完成项或下一步
- Git 提交和推送结果

日志应记录真实执行结果，不使用“能力增强”“持续推进”等模糊表述。

## 8. 验证规范

应尽可能执行与改动直接相关的验证，包括：

- 单元测试
- 脚本烟测
- 服务健康检查
- 转换产物校验
- 文档路径和引用检查
- `git diff --check`

日志中必须写明命令、通过或失败，以及失败原因。

## 9. Git 规范

每个相对完整的开发动作必须提交并推送。

提交信息必须包含任务编号：

```text
P2-001: Analyze nano-vllm integration points
DOC-002: Restructure development documentation
```

提交前必须检查工作区，避免混入无关修改。

默认推送当前分支。当前分支为 `main` 时执行：

```bash
git push origin main
```

如果推送失败，必须在任务日志中记录真实原因。

## 10. 阶段收尾

完成阶段里程碑时：

1. 更新 `docs/planning/roadmap.md`。
2. 更新 `docs/planning/tasks.md`。
3. 新增或更新 `docs/phases/PHASE-XX-*.md`。
4. 清理阶段总结中的过期待办。
5. 如运行方式变化，同步更新 `docs/runbooks/`。

## 11. 硬性要求

- 不把“代码改完但没写日志”视为完成。
- 不把“本地提交但没推送”视为完成。
- 不允许绕过任务编号直接开发。
- 不允许在任务清单中堆积详细流水。
- 不允许覆盖历史日志。
- 如果开发范式发生变化，优先更新本文件和 `docs/index.md`。
