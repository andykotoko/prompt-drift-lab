# 03 实验结果分析（Results Analysis）

**You are here：** `04_results/03_results_analysis_ZH.md`  
**Upstream：** `01_experiment_design/` → `02_prompt_variants/` → `03_evaluation_rules/`  
**Downstream：** `05_summary_and_outlook/`  
**Sidecar：** `06_methodological_addenda_and_controls/`（控制实验/方法选择）、`07_deep_research/`（参考资料）

## 本文档职责（只做什么，不做什么）
本文档负责：**解释结果如何解读**、给出**失败模式/漂移归因**视角。  
本文档不负责：结果文件路径大全/索引。所有“在哪找文件、在哪看表”请以：
- `04_results/README_ZH.md`
为准。

## 结论边界（避免误解）
- **主定量结论**以 **Prompt B（协议版）** 产物为准。
- **Prompt A（探索版）**仅用于**质性对照**与机制线索。
- 除非显式声明“覆盖范围与评测口径完全对齐”，否则 **A/B 不做混合均值或对比统计**。

## 本仓库中 Prompt Drift 的含义
Prompt drift 指：在任务语义不变前提下，仅对提示词的**格式/措辞/约束表达**做微小改动，就出现可重复的行为变化，常见体现在：
- 指令遵循率下降
- 格式/结构崩坏（schema 违约）
- 语义偏离（答非所问/关键信息缺失/冲突）

## 推荐阅读顺序（看结果不迷路）
1) 先到 `04_results/README_ZH.md` 看汇总表与入口（“数字在哪里”）  
2) 再抽样看 raw outputs + judge rationales（“证据长什么样”）  
3) 最后用本文档的失败模式体系做归因与消融（“为什么会这样”）

## A vs B：为什么主实验固定 Prompt B
- **Prompt A（探索版）**：便于早期跑通流程、收集失败样本；约束更松，漂移更容易暴露。
- **Prompt B（协议版）**：结构锚定更强、约束更可执行，利于跨样本对齐与复核。

> “为何选择 B/不选 A”的可复核论证与控制实验口径，请统一写在 `06_methodological_addenda_and_controls/`。

## Invalid 与失败模式（用于诊断，不进入统计）
部分评测样本不进入定量汇总，归档为 **invalid evaluations**，用于：
- 归纳失败模式（failure modes）
- 解释结果不可比的来源
- 推动协议/提示词改进（不改变既有结论）

### 漂移失败模式分类（覆盖核心类型）
A. **Schema/格式错误**：缺字段、顺序错、类型错、额外文本、结构不可解析  
B. **指令偏离**：漏做要求、忽略约束、擅自加步骤  
C. **语义漂移**：答非所问、关键信息缺失、冲突/幻觉  
D. **稳健性/方差问题**：同提示多次运行差异大  
E. **评测投机**：迎合 rubric 关键词但不解决问题

### 建议 flags（用于 invalid 桶统一标注）
- `PROTOCOL_VIOLATION` / `UNPARSABLE_OUTPUT` / `INCOMPLETE_COVERAGE`
- `JUDGE_REFUSAL_OR_EVASION` / `INTERNAL_INCONSISTENCY` / `CONTEXT_MISALIGNMENT`
- `SELF_JUDGING_BIAS`（仅辅助方法；不作为主证据）

## 漂移归因套路（最小可执行）
观察到漂移后按四步写清楚：
1) **定位**：属于 A–E 哪类（可多选）  
2) **假设**：可能机制（歧义/优先级冲突/约束不足/模型偏好）  
3) **消融**：只改一个因素（分隔符/标题锚/负向约束/长度）  
4) **统计**：给频次/比例；若有多次采样再给方差

## 可复现清单（每条结论必须能回指）
你在 `05_summary_and_outlook/` 写的每一条结论，都应能回指到：
- 汇总表的一行（或聚合口径说明）
- 对应 raw outputs
- judged scores + rationale 片段
- prompt 版本 / judge_prompt_id / run config
