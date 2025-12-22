# EVAL_PROTOCOL_v2.md
Prompt Drift Lab — v2.0 评测协议（Multi-Judge + 交叉评审主方法 + 三评审中位数鲁棒性）
【关键约束：评审阶段禁止联网/检索，仅基于被测输出文本】

本协议用于对三生成模型输出进行可复现评测，并通过多评审降低 LLM-as-judge 偏差。
- Primary（主方法）：交叉评审（leave-one-judge-out），避免自评偏置
- Secondary（次方法）：三评审中位数（median-of-3）作为鲁棒性/敏感性分析
- 评审阶段：所有 judge **不得联网/检索**，不得使用外部信息，仅依据上传 PDF 的文本内容做判断。

---

## 1. 术语（导师可读）
首次出现写全称，后续用简称：
- extended thinking（简称 ET）
- web research（简称 WR）
- deep research（简称 DR）

注意：
- 上述模式用于“生成阶段”的实验记录。
- **评审阶段不使用 DR/WR**，评审仅进行“离线文本判分”。

---

## 2. 模型角色定义
### 2.1 生成模型（systems under test）
- ChatGPT-5.2（生成：ET/可能含WR，取决于你的实验设置）
- Gemini 3 Pro（生成：你使用的生成模式以你的 run log 为准）
- Claude Sonnet 4.5（生成：ET）

### 2.2 评审模型（judges，全部离线、不联网）
- Judge-ChatGPT：ChatGPT-5.2（离线判分，不联网）
- Judge-Gemini：Gemini 3 Pro（离线判分，不联网）
- Judge-Claude：Claude Sonnet 4.5（离线判分，不联网）

---

## 3. 实验单元（每个生成模型固定 16 条件）
每个生成模型输出集合由：
- 问题：Q3、Q4（2）
- 提示词版本：baseline / long / weak / conflict（4）
- 触发方式：implicit / explicit（2）
组成

因此每个生成模型：2 × 4 × 2 = 16 份 PDF  
三生成模型总计：48 份 PDF

---

## 4. 盲审规范（控制变量关键）
### 4.1 盲审输入必须去标识化
提交给任一评审时：
- 文件名/正文不得包含：ChatGPT / Gemini / Claude / canvas 等标识词
- 文件名仅保留条件信息（推荐）：
  - `Q3__baseline__implicit.pdf`
  - `Q4__conflict__explicit.pdf`

### 4.2 工程管理可保留可读命名版（但不得用于盲审输入）
仓库内可另存 human_named 版本用于查找，例如：
- `chatgpt__Q3__baseline__implicit.pdf`

若你在盲审阶段已删除模型名前缀，但后续为了 GitHub/VSCode 管理又加回前缀：
- 必须写明：“盲审输入使用去标识化文件；仓库命名为管理便利不参与盲审。”

推荐目录：
- `judge_blind/`：盲审包（匿名命名）
- `human_named/`：管理包（可含模型前缀）
- `scores/`：评审 JSON 输出

---

## 5. 上传限制：10 + 6 两段门控
若平台一次最多上传 10 份 PDF，则 16 份需分两批上传：
- Batch-1：10 份（评审必须只确认收到，不得分析）
- Batch-2：6 份，随后发送“开始评审”（评审才输出最终 JSON）

即使平台允许一次传 16 份，也建议仍按 10+6 执行以保持跨评审一致。

---

## 6. 主评分 Rubric（0–10）
每份输出按 5 个维度打分（0/1/2），总分 0–10。

A_structure：是否按三段式输出且顺序正确  
B_snapshot_constraint：第1部分是否 ≤50字且仅陈述现象/结论（禁分析/建议）  
C_actionability：第2/3部分是否为可执行研究指令（动词 + 产物要求）  
D_completeness：Step2 是否要求时间戳/来源；Step3 是否要求机制/分歧/证据对照  
E_drift_failure：缺段/乱序/回退普通回答/忽略结构等漂移失败（无=2，轻微=1，严重=0）

total = A+B+C+D+E

---

## 7. 探索性一致性（Authorship-like, 非主结论）
先标注结构模式：
- T（Template）：严格三段式模板
- A（Analysis）：解释/分析模式（非三段式）

输出：
- same_system_probability_within_mode：分别在 T / A 内估计同一系统概率（0-100）
- same_system_probability_overall_mixed：总体混合概率（模式混合会降低，这是预期现象）
- across_versions：四版本差异更像提示词差异还是系统差异（探索性）

---

## 8. 评审输出格式：必须只输出 1 个 JSON
每次评审一个 bundle（16 份）只输出一个 JSON，不得附加解释文本或 Markdown。

JSON Schema（字段名与层级不得改动）：

{
  "bundle_meta": {
    "bundle_size": 16,
    "questions": ["Q3","Q4"],
    "versions": ["baseline","long","weak","conflict"],
    "trigger_types": ["implicit","explicit"]
  },
  "mode_labels": [
    {"file":"", "mode":"T/A", "evidence":"<=25字引用"}
  ],
  "per_file_scores": [
    {
      "file":"",
      "scores":{
        "A_structure":0,
        "B_snapshot_constraint":0,
        "C_actionability":0,
        "D_completeness":0,
        "E_drift_failure":0
      },
      "total":0,
      "evidence":{
        "A_structure":"",
        "B_snapshot_constraint":"",
        "C_actionability":"",
        "D_completeness":"",
        "E_drift_failure":""
      },
      "notes":"一句话"
    }
  ],
  "aggregates": {
    "avg_total": 0,
    "implicit_vs_explicit_summary": {
      "implicit":{"avg_total":0,"notes":""},
      "explicit":{"avg_total":0,"notes":""}
    },
    "version_level_summary": {
      "baseline":{"avg_total":0,"notes":""},
      "long":{"avg_total":0,"notes":""},
      "weak":{"avg_total":0,"notes":""},
      "conflict":{"avg_total":0,"notes":""}
    },
    "same_system_probability_within_mode": {"T":0,"A":0},
    "same_system_probability_overall_mixed": 0,
    "num_clusters_suggested": 0,
    "cluster_summary": [
      {"cluster_id":"C1","mode":"T/A","files":[],"rationale":"一句话"}
    ],
    "across_versions": {
      "same_system_across_versions_probability": 0,
      "interpretation": "更像提示词差异/更像系统差异/不确定",
      "version_fingerprints": {
        "baseline":{"fingerprint":"","evidence":[]},
        "long":{"fingerprint":"","evidence":[]},
        "weak":{"fingerprint":"","evidence":[]},
        "conflict":{"fingerprint":"","evidence":[]}
      }
    }
  },
  "final_notes":"必须写出任何异常与限制（例如Q4内容错位）"
}

---

## 9. 聚合规则（Aggregation）
### 9.1 Primary：交叉评审（leave-one-judge-out）
对每个生成模型的 bundle（16 份）：
- 使用另外两位评审的 JSON（排除自评）
- 对每个样本最终分数：final_total = mean(total_j1, total_j2)
- 若两位评审 mode 冲突，标记 mode_disagree（后续少量人工抽查即可）

### 9.2 Secondary：三评审中位数（median-of-3）
鲁棒性分析：
- 三位评审都评同一 bundle（含自评）
- 每个样本取 median(total_chatgpt, total_gemini, total_claude)
- 若主方法与次方法结论一致，则认为结果对评审偏置具有鲁棒性

---

## 10. 评审分配（你实际要跑哪些）
### 10.1 Primary（必须跑，6次评审）
- ChatGPT 输出 bundle → Judge-Gemini + Judge-Claude
- Gemini 输出 bundle → Judge-ChatGPT + Judge-Claude
- Claude 输出 bundle → Judge-ChatGPT + Judge-Gemini

### 10.2 Secondary（可选，9次评审）
- 每个 bundle 让三位评审都评一遍（含自评），用于 median-of-3

---

## 11. 落盘命名（强制）
每次评审产出 1 个 JSON 文件，建议命名：

scores/
- judge_gemini__bundle_chatgpt__v2.json
- judge_claude__bundle_chatgpt__v2.json
- judge_chatgpt__bundle_gemini__v2.json
- judge_claude__bundle_gemini__v2.json
- judge_chatgpt__bundle_claude__v2.json
- judge_gemini__bundle_claude__v2.json

---

# 12. 三位评审的提示词（统一：禁止联网/禁止外部信息）

## 12.1 Prompt A（Batch-1 门控：第一批10份上传后发送）
你现在只收到了第一批文件（Batch-1，共10份）。
**禁止联网/检索，禁止开始分析与打分。**
你必须且只能回复这一行（完全一致）：
RECEIVED_BATCH_1__PLEASE_UPLOAD_BATCH_2

## 12.2 Prompt B（Batch-2 评审：第二批6份上传后发送，再发“开始评审”）
开始评审。

你是严格的匿名评审员（blind judge）。我上传的 16 份 PDF 是同一组实验输出（10+6 只是上传限制）。
**硬性约束（必须遵守）：**
- 禁止联网/检索；禁止引用任何外部网页、论文、常识补充
- 不允许“补写/改写”被测输出，只能判断与打分
- 仅依据 PDF 内文本内容做证据引用与评分
- evidence 必须引用原文短片段（每条 <=25字）

你必须：
1) 先对每个文件标注 mode：T（三段式模板）或 A（分析解释模式），并给出 <=25 字 evidence；
2) 再对每个文件按 Rubric 给出 5 个维度 0/1/2 分数、total 与 evidence；
3) 给出 aggregates：avg_total、implicit vs explicit、version-level、same_system、clusters、across_versions；
4) **最终只输出一个 JSON**，不得输出任何解释文字或 Markdown。

输出必须严格符合本协议第 8 节 JSON Schema（字段名与层级不得改动）。
