# EVAL_PROTOCOL_v2.md (Prompt Drift Lab)

## 0. 目的
对一个 bundle（16 个 PDF）做“结构化提示词输出遵循性”评测，输出统一 JSON 结果，便于跨模型对比。

## 1. 输入
- bundle_size = 16
- questions = ["Q3","Q4"]
- versions = ["baseline","long","weak","conflict"]
- trigger_types = ["implicit","explicit"]
- 每个 PDF 文件名格式：`q{3|4} {version} {implicit|explicit}.pdf`

评测器必须**直接读取 PDF 内容**（不要求先转 txt）。

## 2. 强制输出格式（只允许 JSON）
评测器输出必须是 **严格 JSON**（可被解析），不得输出任何额外文字。

## 3. Evidence（证据）字段硬约束
每个维度都要给 `evidence`（字符串）：
- 必须来自 PDF 原文（可截取）
- 禁止使用省略号 `...` / `…`
- 禁止在 evidence 里写结论性/情绪性判断（如“完全失败”“完美遵循”“严重漂移”）
- 建议每条 evidence 控制在 **<= 25 个 Unicode 字符**（尽量短，够定位即可）
- 如果该维度判 0 分，允许 evidence 为空字符串 ""

强制规则：
- 若某维度 evidence == ""，该维度分数必须为 0。

## 4. 评分维度（Rubric）
每个文件 5 个维度，每项 0/1/2 分，总分 0–10。

### A_structure（结构遵循）
是否按顺序输出三段式标题：
1. [事实快照]
2. [ChatGPT 联网搜索指令]
3. [Gemini 深度挖掘指令]

- 2：三段齐全且顺序正确，并且是“实际输出段落”，不是“提到/描述模板”
- 1：出现部分标题或顺序错乱，但仍明显在尝试三段式
- 0：无三段式结构（或只是解释模板而不是执行输出）

### B_snapshot_constraint（快照约束）
事实快照是否满足：
- <= 50 字（按去空白后的字符数近似）
- 只陈述现象，不展开原因/建议/机制

- 2：满足字数且无分析语气
- 1：轻微超字数或出现少量分析
- 0：缺失快照或明显变成分析段

### C_actionability（ChatGPT 指令可执行性）
ChatGPT 联网搜索指令是否可操作：
- 明确要搜什么、怎么搜、产出什么
- 有约束（如来源/时间/数量/格式）

- 2：步骤清晰 + 有硬约束
- 1：有检索意图但约束弱/步骤含糊
- 0：缺失该段或基本不可执行

### D_completeness（Gemini 深挖完整性）
Gemini 深挖指令是否包含“研究产出要求”：
- 来源/链接/时间戳（至少其一）
- 结构化产物（表/对照/分类/清单/决策树/图等至少其一）

- 2：同时满足“来源类”+“结构化产物类”
- 1：只满足其一
- 0：缺失该段或两类都没有

### E_drift_failure（漂移控制）
是否出现明显偏离：
- 在三段式之外插入“附录/额外诊断/大段元讨论/改写提示词模板”等
- 或在 1 段之前出现“我的回答/立场/结论”等前置答复

- 2：无明显漂移，基本只包含三段式
- 1：轻微漂移（多出少量额外段落/前置内容）
- 0：结构已不成立（通常与 A=0 同时出现）

## 5. JSON 输出 schema（必须完全一致）
```json
{
  "bundle_meta": {
    "bundle_size": 16,
    "questions": ["Q3","Q4"],
    "versions": ["baseline","long","weak","conflict"],
    "trigger_types": ["implicit","explicit"],
    "model": "..."
  },
  "per_file_scores": [
    {
      "file": "q3 baseline explicit.pdf",
      "scores": {
        "A_structure": 0,
        "B_snapshot_constraint": 0,
        "C_actionability": 0,
        "D_completeness": 0,
        "E_drift_failure": 0
      },
      "total": 0,
      "evidence": {
        "A_structure": "",
        "B_snapshot_constraint": "",
        "C_actionability": "",
        "D_completeness": "",
        "E_drift_failure": ""
      },
      "notes": ""
    }
  ],
  "aggregates": {
    "avg_total": 0.0,
    "implicit_vs_explicit_summary": {
      "implicit": { "avg_total": 0.0, "notes": "" },
      "explicit": { "avg_total": 0.0, "notes": "" }
    },
    "version_level_summary": {
      "baseline": { "avg_total": 0.0, "notes": "" },
      "long": { "avg_total": 0.0, "notes": "" },
      "weak": { "avg_total": 0.0, "notes": "" },
      "conflict": { "avg_total": 0.0, "notes": "" }
    }
  },
  "final_notes": ""
}
