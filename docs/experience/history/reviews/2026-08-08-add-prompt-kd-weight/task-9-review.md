> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-9-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9 Review: Documentation

**Reviewer:** documentation review (Task 9)  
**Date:** 2026-08-10  
**Artifacts:** `task-9-brief.md`, `task-9-report.md`, `task-9-review-package.diff`  
**Code cross-check:** `train_utils/distill_losses.py` (`build_distill_token_mask`), `train_utils/cat_train_args.py`, `compressed_e2e_fintuning/args.py`

---

## Spec Checklist

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | `docs/cat_train_args.md` 增加 category 参数：`default=0.0`、范围 `>=0`、支持 after-category override | ✅ | §2.2 override 列表 + 示例；§3 参数表一行；§6.11.1 专节 |
| 2 | `compressed_e2e_fintuning/README.md` 增加 Prompt KD weighting 小节 | ✅ | Hidden-state 对齐之后新增 `## Prompt KD weighting` |
| 3 | 明确 `0.0` / `0.05` / `1.0` 语义 | ✅ | 两处均有 bullet/表格；与 `build_distill_token_mask` 行为一致 |
| 4 | 明确该权重不改变 CE 和 hidden loss | ✅ | README 第 27 行；cat_train_args 参数表 + §6.11.1 首段；代码中 CE 走 `_causal_lm_cross_entropy`，hidden 走独立路径，mask 仅传入 distill loss |
| 5 | 明确 padding / final logits 排除，EOS target 仍包含 | ✅ | 两处 mask 规则；§6.11.1 逐条列出；与 causal shift（`source_weights[:, 1:]` → `causal_mask[:, :-1]`）及 `labels != -100` 逻辑一致 |
| 6 | 明确 EAKLD entropy/gamma 与 KL 共用同一 weighted mask | ✅ | 两处均写明；trainer 中 `gamma_mask = self._build_distill_token_mask(...)` 与 KD `token_mask` 同源 |
| 7 | `0.05` / `0.1` 仅作实验示例，不写成推荐最优或已验证结论 | ✅ | 两处均有「以下数值仅作实验示例，**不是**推荐最优值或已验证结论」及对应 CLI 示例 |

**Spec verdict: ✅ 全部满足**

---

## Quality

### Strengths

- **双入口覆盖完整**：cat LoRA 路径（`--distill_prompt_kd_weight`）与 e2e 路径（`--prompt_kd_weight`）各自文档化，参数名与各自 CLI 一致。
- **§6.11.1 深度足够**：补充了 target→logits 左移因果语义、逐类权重规则、EAKLD 归一化公式、`>1.0` 实验允许说明——超出 brief 最低要求，且与实现吻合。
- **边界声明准确**：CE / hidden / pre-MLP hidden 不受影响；mcqa 不支持 `!= 0`（e2e README 额外说明，与 `args.py` 校验一致）。
- **与代码一致**：report 声称对照 `build_distill_token_mask` 与 `_DISTILL_PROMPT_KD_WEIGHT_SPEC`，diff 内容与当前工作区文件一致，未发现语义偏差。

### Minor Issues (non-blocking)

1. **参数表行过长**（`docs/cat_train_args.md` §3 `--distill_prompt_kd_weight` 行）：单 cell 塞入全部语义，可读性偏弱；细节已在 §6.11.1 展开，表行可酌情缩短为「见 §6.11.1」。
2. **两文档深度不对称**：e2e README 未写 target→logits 左移与 `>1.0` 说明；对 e2e 用户通常够用，若希望两文档对等可后续补一句。
3. **无交叉引用**：cat 与 e2e 文档未互链；各自独立可用，但维护时需同步改两处。

### Not in Scope / Acceptable Omissions

- 未运行测试：文档-only 变更，report 已说明，可接受。
- §6.11.1 提及 pre-MLP hidden alignment，e2e README 未提：e2e 路径可能无此参数，不构成 spec 缺失。

---

## Overall

| Dimension | Verdict |
|---|---|
| **Spec** | ✅ Pass — 7/7 必填语义均已文档化 |
| **Quality** | **Good** — 内容准确、与实现一致；§6.11.1 质量高；仅有可读性/对称性上的小改进空间 |

**Recommendation:** 可合并。无需返工；若 polish，优先缩短 §3 参数表长行并在 e2e README 加一句「权重在 target 位置定义、左移到 logits」。
