# 经验三级分类与分类检索整理记录

日期：2026-11-09。范围：现有可复用经验分类、检索工具及文档维护操作规范。用户确认按三级分类整理，并明确要求保留所有信息及可能的并行总结；同时要求按分类快速检索。

## 组织与入口

- 唯一分类导航：[经验库](../../README.md#按问题查找)。一级为八个阶段／通用领域目录，二级为30个问题专题文件，三级为经验条目。
- 从10篇旧经验正文提取286个内容块（包括原导航、上下文与边界，并非286条独立实证结论）。保留原有日期、数值、限制、假设、负例、来源、代码片段和正文；只调整归属、标题层级和相对链接，并为拆分的问题添加定位标题。
- 旧文件保留原标题锚点及按原顺序排列的新去向；旧经验正文不维护第二份。实验记录、历史记录、协议和运行产物保持原位置；当前计划仅记录本次文档交付入口。
- 仓库原 `data/` 忽略规则也覆盖了数据经验目录，已仅对 `docs/experience/lessons/data/*.md` 放行；数据集及该目录非 Markdown 文件继续忽略，防止后续版本保存遗漏这一分类。
- `AGENTS.md` 维护分类、检索、增量整理和并发保护义务；分类导航在经验库，条目模板在 `WORKFLOW.md`。本次不定义算法流程、变更验收口径或新增实验授权。

## 分类检索

在已激活的 `bitvae` 中：

```bash
python tools/docs.py categories
python tools/docs.py categories --category category_recovery
python tools/docs.py search '学习率|步数' --category category_recovery --topic optimization
python tools/docs.py search 'mask|截断' --category 数据与监督 --topic supervision
python tools/docs.py search '翻码|proxy' --scope lessons
```

分类接受英文键或导航中的中文名，专题使用文件名（不含 `.md`）。分类检索只遍历指定分类目录；旧命令与 scope、正则、limit 保留。旧跳转索引不重复进入正文检索，但索引区外并行追加的正文保留真实行号、可由全经验检索找到。未知分类／专题、缺分类的专题过滤及与 records/history 的混用明确报错。

## 原文保留与并发核对

迁移前保存最新文件快照及散列，不以 Git HEAD 或暂存版本代替当前工作区。分类草稿先按原顺序重组全部内容块，要求与10篇原文逐字相同；落盘后逐块核对转换后的正文及锚点，比较缺失链接的解析后目标。新正文先落盘，再逐文件重读比对旧正文并原子替换为导航，防止旧快照覆盖期间的新总结。收尾复查旧入口与预期导航，未发现需归并的并发补写；该结论只覆盖本次核对时点，不保证未来无人继续补写。

原文总计638463字节；下列散列对应迁移前工作区的 UTF-8 原文，包含此前未提交更新。旧文件链接可追溯全部内容块的新去向。

| 原专题 | 原文字节 | 内容块数 | 迁移前 SHA-256 |
| --- | ---: | ---: | --- |
| [checkpoint_lifecycle.md](../../lessons/checkpoint_lifecycle.md) | 44275 | 25 | `74f530b0887921727b5f38e08773485719d9226df508a2483ef57e57d380b849` |
| [compression_choices.md](../../lessons/compression_choices.md) | 158543 | 56 | `7fb275a4cff7356092090be0d3fbc1f415e91b06189fb5cc4b30f196ade90aba` |
| [experiment_design.md](../../lessons/experiment_design.md) | 51465 | 25 | `9e45b02c02d81bacd67ad018260814e68e181ed130889aa2f5ddefb248b9b674` |
| [kd_and_data.md](../../lessons/kd_and_data.md) | 124534 | 49 | `c5acf04f79dcb2d084793f7747d4f05b5b6e020c3d0626848c18df5010098998` |
| [liftquant_recovery.md](../../lessons/liftquant_recovery.md) | 109181 | 46 | `593be8db2391bde6eaba08926811c07017fd654042d5cb55479b4f42319ccda3` |
| [memory_precision.md](../../lessons/memory_precision.md) | 54649 | 31 | `b099587a04711f21398b8a7bb1aa092947d5a7aabac4e35e41940cacb98dde36` |
| [mixed_bit_pipeline.md](../../lessons/mixed_bit_pipeline.md) | 9157 | 8 | `b2113025a2770b70822e7e6854ef2d2c4bf2d98aa9caeec22e2f8479e85503e0` |
| [output_losses.md](../../lessons/output_losses.md) | 47496 | 20 | `c535eef0147c44ab47549ce2d9692251e1f7047ec5002382365e5fc96fe3fe23` |
| [reproducibility.md](../../lessons/reproducibility.md) | 26109 | 15 | `5a59e9a565ace3d3da5a6238959d30ce7bd3d517aa484b9df71d5c77a08fa7fa` |
| [residual_lora.md](../../lessons/residual_lora.md) | 13054 | 11 | `eef3223102e35bb768445cb3b0f62d3b5c22dbc3ed203693e59bd5f789485952` |

## 验证与限制

- 原文按序还原、286块落盘内容、原标题和显式锚点保留：通过；新增专题及旧入口的相对锚点检查无新增失败。
- `bitvae` Python 3.11.16；8项分类检索回归通过，覆盖分类隔离、中文名、专题、旧 scope、分类列举、非法参数、正则／limit，以及旧入口前后并发补写的检索和真实行号。
- 索引已刷新且 `index --check` 通过；全库迁移前后均为1654处既有相对路径缺失，解析后的缺失目标无新增、原缺失引用无丢失；本次受影响文档的锚点检查无新增失败。已有断链不因搬移而视为已修复，也不据此删除证据引用。
- 本次未训练、评测或修改算法与正式运行配置；分类不升级任何技术经验的证据状态。

## 经验增量与收尾

无新增算法／质量经验；新增的是分类维护、分类检索及并发总结保护的操作要求，已写入 [AGENTS.md](../../../../AGENTS.md#文档检索与经验管理)。复用条目仍保留其原适用条件和证据。

保留本记录、分类正文、旧锚点入口、检索工具及长期回归。最终核对后清理本次临时目录 `/tmp/vaellm_lessons_20261109_aqy0fx9f`（67个文件，4255776字节，4.059 MiB），内容为一次性迁移脚本、已核对快照、暂存草稿及诊断日志；唯一核对结论和原文散列保存在本记录，不触及实验权重、原始数据和运行总结。

检索实现 SHA-256：`5e13dfefc91461d965ebb6479618b0bf7cf2b6e579f9b16c4501d4b1e2eb1326`；回归 SHA-256：`08f25b66bc40daafe699fc2ed133a0ac1094f759366d38eda4319ff0f573be34`。工作区存在其他未提交改动，本次不暂存或提交。
