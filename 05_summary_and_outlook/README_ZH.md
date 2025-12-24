# 05 总结与展望（Summary & Outlook）

本目录是仓库的**解释层（interpretive layer）**：只做“基于已落盘结果的总结、边界澄清、未来方向记录”，不引入超出现有证据的新结论。

> 说明：本版本仓库**不包含**自动化扰动生成器（perturb）/ 大规模 prompt 扫描系统。除非仓库中存在可运行工件，否则不得暗示已实现。

---

## 0) 30 秒导航

- 想看证据（原始输出 / 评测 JSON / 汇总 CSV）→ `04_results/`
- 想看评测契约与评分规则 → `03_evaluation_rules/`（`EVAL_PROTOCOL.md`、`JUDGE_PROMPT.md`）
- 想看生成侧提示词与变体 → `02_prompt_variants/`（`PROMPT_MANIFEST.md`）

---

## 1) Claim 边界（哪些能说 / 不能说）

### ✅ 可以说（必须可追溯）
你可以：
- 总结**本仓库已保存输出**中观察到的现象（prompt 变体 × 模型 × 题目）。
- 报告**直接由已保存表格**计算得到的汇总结果（`04_results/**/summary_tables/`）。
- 基于评测输出描述**失败模式**（如 schema/格式崩坏、指令偏离），并提供证据指针。
- 明确本实验设置的**局限与非覆盖范围**。
- 提出**假设/未来工作**（必须标注为“假设”，不能写成“已发现/已证明”）。

### ❌ 不能说（非结论）
不得声称：
- 泛化性的模型能力结论、或“模型 X 优于模型 Y”的通用判断（超出本项目设置即不成立）。
- 真实世界 safety/alignment 改善（除非本仓库明确测量并提供证据）。
- benchmark 级别结论、跨领域外推、外部有效性（除非证据覆盖）。
- 仓库未实现/未落盘的新指标、新维度、新实验或新系统。

---

## 2) 可追溯规则（硬约束）

本目录所有“结果性表述”必须给出**证据文件路径**，例如：

- 原始输出：`04_results/01_raw_model_outputs/...`
- 评测 JSON：`04_results/02_cross_model_evaluation/...`
- 汇总 CSV：`04_results/**/summary_tables/*.csv`
- 协议快照（若存在）：`04_results/**/used_evaluation_protocol*.md`

**没有路径指针的句子**：要么改写成“假设/猜测”，要么删除。

---

## 3) 分工边界（该放什么 / 不该放什么）

### 属于 `05_summary_and_outlook/`
- 基于 `04_results/` 的高层总结（带证据指针）
- 方法层面的含义（这个设置擅长/不擅长什么）
- 明确的 non-claims 与局限
- 未来方向（标注为 future work）

### 不属于本目录
- 新评分规则/评测逻辑 → `03_evaluation_rules/`
- 提示词正文/变体文本 → `02_prompt_variants/`
- 主结果工件（JSON/CSV/PDF）→ `04_results/`