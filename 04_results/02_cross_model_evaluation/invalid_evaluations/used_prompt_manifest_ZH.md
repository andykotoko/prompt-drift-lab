# 使用的提示词清单（Judge Prompts Inventory）

本文档用于**逐条记录** Prompt Drift Lab 项目在**评测阶段**实际使用的提示词（judge prompts），以保证：

- **可追溯**：任意一条评测结果都可以明确追溯到“使用了哪一条 judge 提示词”；
- **可复现**：他人能够在相同评测协议与提示词条件下复跑实验并获得可比结果；
- **可审计**：当评测结果被判定为无效（invalid）时，可以明确区分是“评测协议执行失败”，还是“评测提示词设计本身导致”。

> **说明**：本清单**仅列出评测阶段**使用的提示词（judge prompts），
> 不包含任何生成阶段（generation stage）的提示词；
> 生成阶段的提示词请在对应目录或文档中单独记录。

---

## 0. 命名与引用约定（Naming & Referencing Conventions）

- 每一条评测提示词均使用一个**稳定且唯一的 ID**，格式为：`JP-XXX`（Judge Prompt）。
- 每条提示词必须显式声明以下元信息字段：
  - `method`：评测方式，取值为 `cross_judge` 或 `self_judge`；
  - `protocol_file`：对应使用的评测协议文件名（位于本目录内）；
  - `expected_output`：评测输出的严格格式要求（本项目中统一为 **单一 JSON 对象**）；
  - `scope`：该提示词的适用范围（例如：所有题目 / 特定 question\_id / 特定输出类型）。

在评测结果 JSON 的 `meta` 字段中，**强烈建议**加入以下键值：

- `judge_prompt_id`: `JP-001`
- （可选）`protocol_file`: `EVAL_PROTOCOL_ZH.md`

---

## 1. 主方法：跨模型互评（Cross-model Judging）

### JP-001（Cross Judge / 通用）

- **method**：`cross_judge`
- **protocol_file**：`./EVAL_PROTOCOL_ZH.md`
- **expected\_output**：仅输出一个 JSON 对象（禁止任何 Markdown、解释性段落或前后缀文本）
- **scope**：适用于所有 `question_id`、所有 `prompt_variant`、所有 `target_model`

**Prompt（全文）**：

```
你是一个严格的评测模型（judge）。
你将接收：
(1) 被评测模型的输出文本；
(2) 与该输出对应的元信息（question_id、prompt_variant、target_model、output_id）。

你的任务：
1) 严格按照《EVAL_PROTOCOL_ZH.md》对该输出进行评测：
   - 维度：FORMAT_COMPLIANCE / INSTRUCTION_COMPLIANCE / SEMANTIC_FIDELITY / COMPLETENESS
   - 每项打分：0 / 1 / 2（整数）
   - overall_score = 四项分数之和
2) 给出 verdict：PASS / PARTIAL / FAIL（按照评测协议中定义的阈值）
3) 对每个评测维度提供至少 1 条 evidence（原文 quote + 1 句理由说明）
4) 若出现以下情况之一：
   - 评测协议未遵循
   - 输出不可解析
   - 覆盖不完整
   - 拒评 / 回避
   - 内部不一致
   则必须在 flags 中填入对应标签：
   PROTOCOL_VIOLATION / UNPARSABLE_OUTPUT / INCOMPLETE_COVERAGE /
   JUDGE_REFUSAL_OR_EVASION / INTERNAL_INCONSISTENCY

输出要求（非常重要）：
- 你必须且只能输出 **一个 JSON 对象**，不得包含任何 Markdown、解释段落、前后缀或代码块标记。
- JSON 字段必须齐全：meta / scores / verdict / flags / evidence / notes。
- scores 中四个维度必须全部出现，且取值必须为 0 / 1 / 2 的整数。

现在开始评测。输入如下：
[meta]
{META_JSON}

[target_output]
{TARGET_OUTPUT}
```

---

## 2. 辅助方法：模型自评（Self Judging）

### JP-002（Self Judge / 通用）

- **method**：`self_judge`
- **protocol\_file**：`./EVAL_PROTOCOL_ZH.md`
- **expected\_output**：仅输出一个 JSON 对象（禁止任何 Markdown、解释性段落或前后缀文本）
- **scope**：适用于所有 `question_id`、所有 `prompt_variant`、所有 `target_model`

**Prompt（全文）**：

```
你是一个严格的评测模型（judge）。
你将对“你自己先前生成的输出”进行评测。

你将接收：
(1) 你的输出文本；
(2) 与该输出对应的元信息（question_id、prompt_variant、target_model、output_id）。

你的任务：
1) 严格按照《EVAL_PROTOCOL_ZH.mdFORMAT_COMPLIANCE / INSTRUCTION_COMPLIANCE / SEMANTIC_FIDELITY / COMPLETENESS
   - 每项打分：0 / 1 / 2（整数）
   - overall_score = 四项分数之和
2) 给出 verdict：PASS / PARTIAL / FAIL（按照评测协议中定义的阈值）
3) 对每个评测维度提供至少 1 条 evidence（原文 quote + 1 句理由说明）
4) 若出现以下情况之一：
   - 评测协议未遵循
   - 输出不可解析
   - 覆盖不完整
   - 拒评 / 回避
   - 内部不一致
   则必须在 flags 中填入对应标签：
   PROTOCOL_VIOLATION / UNPARSABLE_OUTPUT / INCOMPLETE_COVERAGE /
   JUDGE_REFUSAL_OR_EVASION / INTERNAL_INCONSISTENCY

输出要求（非常重要）：
- 你必须且只能输出 **一个 JSON 对象**，不得包含任何 Markdown、解释段落、前后缀或代码块标记。
- JSON 字段必须齐全：meta / scores / verdict / flags / evidence / notes。
- scores 中四个维度必须全部出现，且取值必须为 0 / 1 / 2 的整数。

现在开始评测。输入如下：
[meta]
{META_JSON}

[self_output]
{TARGET_OUTPUT}
```

---

## 3. 变量与元信息模板（供直接粘贴使用）

```json
{
  "judge_model": "...",
  "target_model": "...",
  "question_id": "Q1",
  "prompt_variant": "A",
  "output_id": "...",
  "method": "cross_judge",
  "judge_prompt_id": "JP-001",
  "protocol_file": "EVAL_PROTOCOL_v2.mdEVAL_PROTOCOL_ZH.md- `judge_prompt_id` 必须与本清单中定义的 `JP-XXX` 严格对应。
- 可选扩展字段包括（但不限于）：`run_id`、`seed`、`temperature`、`top_p` 等，用于复现实验条件。

