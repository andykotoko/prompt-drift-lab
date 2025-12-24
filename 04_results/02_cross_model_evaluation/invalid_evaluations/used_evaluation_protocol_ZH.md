# 使用的评测协议（Judge Protocol）

本文档定义本项目在 **跨模型互评** 与 **模型自评** 中采用的统一评测协议。
目标是让评测结果 **可解析、可对齐、可复核**，并能明确地区分“有效评测”与“无效评测”。

---

## 0. 评测对象与单位

- **评测对象**：模型在同一题目（Question）下、使用不同提示词版本（Prompt Variant）产生的输出。
- **评测单位（sample）**：
  - `question_id`（例如 Q1–Q4）
  - `prompt_variant`（例如 A/B，或 long/short、explicit/implicit 等你仓库中约定的命名）
  - `target_model`（被评测输出来自哪个模型）
  - `output_id`（文件名或唯一标识，用于追溯到原始输出文件）

> 说明：本协议只规范“如何评测与如何产出可解析结果”，不规定你如何设计题目与提示词。

---

## 1. 评测方法

### 1.1 主方法：跨模型互评（Cross-model Judging）
- 评测者模型（judge）对其他模型的输出进行打分与标签标注。
- 目的：更贴近第三方评审/用户视角，降低自我偏置。

### 1.2 辅助方法：模型自评（Self-judging）
- 模型对自身输出进行同样协议的评测。
- 目的：用于对照、排查协议执行问题、补充失败模式线索。
- 限制：自评结果 **不作为主统计依据**（除非你在结果分析章中单独声明其用途）。

---

## 2. 评测维度（Rubric）

本项目关注“提示词微小变化导致的遵循率下降、格式崩坏、语义漂移”等现象。为便于跨模型对齐，采用以下通用维度：

### D1. 结构/格式遵循（FORMAT_COMPLIANCE）
- 是否按要求的结构输出（如 Markdown/JSON 模板、分段、标题层级、字段齐全等）。

### D2. 指令遵循（INSTRUCTION_COMPLIANCE）
- 是否遵循关键约束（例如：必须三段式、必须包含某些字段、必须避免某些内容、必须引用证据等）。

### D3. 语义一致性/漂移（SEMANTIC_FIDELITY）
- 与题目意图/提示词目标的贴合度；是否出现偏题、偷换任务、答非所问、泛化成通用建议等。

### D4. 完整性（COMPLETENESS）
- 是否覆盖所有必答要点；是否缺项/跳项/只答一部分。

> 可选：若你已有更细的 8 项指标，可将其映射到以上 4 个维度（例如“格式崩坏”→D1，“遵循率”→D2，“语义漂移”→D3，“遗漏字段/漏答”→D4），在统计表中做展开。

---

## 3. 评分标尺

每个维度采用 **0–2** 的离散评分（便于减少主观噪声）：

- **2 = 满足**：清晰满足该维度要求；即便有小瑕疵也不影响解析/复核。
- **1 = 部分满足**：存在明显偏离，但仍可识别其意图/部分结构仍可用。
- **0 = 不满足**：该维度失效（例如结构完全崩坏、指令关键约束被忽略、语义显著跑偏、严重漏答）。

同时给出：
- `overall_score`：四个维度分数之和（0–8）。
- `verdict`：`PASS` / `PARTIAL` / `FAIL`（可按阈值定义，见下）。

### 建议阈值（可在统计时统一使用）
- `PASS`：overall ≥ 7 且 D1、D2 均不为 0
- `PARTIAL`：overall 4–6 或存在单项 0
- `FAIL`：overall ≤ 3

> 阈值不是“真理”，但应固定在一次实验周期内保持一致。

---

## 4. 证据要求（Evidence）

评测输出必须包含 **可复核的证据片段**，以支持分数与标签：

- 每个维度至少给出 **1 条证据**（可复用同一片段）。
- 证据形式：
  - `quote`：从被评测输出中截取的短片段（建议 ≤ 30–60 字/词，避免长引用）
  - `reason`：该片段为何支持该维度分数（1 句即可）

> 如果原始输出是 PDF/长文本，证据只需“能定位与核对”，不强求行号。

---

## 5. 输出格式（严格 JSON）

评测者必须输出 **仅一个 JSON 对象**，不允许 Markdown、解释性段落或额外前后缀。推荐结构如下：

```json
{
  "meta": {
    "judge_model": "...",
    "target_model": "...",
    "question_id": "Q3",
    "prompt_variant": "A",
    "output_id": "...",
    "method": "cross_judge",
    "timestamp": "YYYY-MM-DD"
  },
  "scores": {
    "FORMAT_COMPLIANCE": 0,
    "INSTRUCTION_COMPLIANCE": 0,
    "SEMANTIC_FIDELITY": 0,
    "COMPLETENESS": 0,
    "overall_score": 0
  },
  "verdict": "FAIL",
  "flags": [
    "PROTOCOL_VIOLATION",
    "UNPARSABLE_OUTPUT"
  ],
  "evidence": [
    {"dimension": "FORMAT_COMPLIANCE", "quote": "...", "reason": "..."}
  ],
  "notes": "(可选)"
}
```

### 字段约束
- `scores` 中四个维度必须都有，且为整数 0/1/2。
- `overall_score` 必须等于四维度之和。
- `flags` 可为空数组 `[]`，但不可缺省。
- `method` 只能是：`cross_judge` 或 `self_judge`。

---

## 6. 无效评测判定规则（Invalid Criteria）

出现以下任一情况，该条评测结果应归入 `无效评测/`，并在 `flags` 中标注对应标签：

- **PROTOCOL_VIOLATION**：未按本协议输出 JSON，或维度/标尺被更改。
- **UNPARSABLE_OUTPUT**：JSON 语法错误、字段缺失、类型不匹配、无法解析。
- **INCOMPLETE_COVERAGE**：样本信息不完整（缺 `question_id`/`prompt_variant`/`target_model`/`output_id` 等关键字段），或评测集合缺项导致无法对齐。
- **JUDGE_REFUSAL_OR_EVASION**：拒绝评测/不给分/输出与任务无关内容。
- **INTERNAL_INCONSISTENCY**：例如 `overall_score` 不等于分项之和、同一维度给出矛盾分数且无法最小修复。

> 注意：无效评测是“不可用结果”，不是“低分结果”。低分只属于 `FAIL`，不属于 invalid。

---

## 7. 执行流程（推荐固定）

1. **确定样本集合**：列出本轮应评测的所有 `output_id`（按 question × variant × target_model 对齐）。
2. **调用评测者（judge）**：对每个样本执行一次评测，强制输出 JSON。
3. **结构校验**：
   - JSON 语法是否正确
   - 字段是否齐全
   - 分数是否在 0/1/2
   - `overall_score` 是否一致
4. **归档**：
   - 校验通过 → `有效评测/...
   - 校验失败或满足无效条件 → `无效评测/...`（并记录 `flags`）

---

## 8. 复现信息（建议记录在运行日志中）

为保证复现，建议在每次批量评测的 run log 中记录：
- judge 模型与版本（如可得）
- 解码参数（temperature、top_p、max_tokens 等）
- 是否启用工具/联网（如有）
- 评测时间范围与样本数量

---

## 9. 与其他文档的关系

- `使用的提示词清单.md`：列出本轮评测实际使用的 judge 提示词（含 cross/self）。
- `无效评测/README.md`：说明 invalid 的边界与目录角色。
- `04_实验结果/03_实验结果分析.md`：在“事实已固定”的前提下讨论失败模式与机制解释。
