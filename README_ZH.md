# Prompt Drift Lab
*面向可复现的提示词漂移（prompt drift）评测框架：用受控提示词变体检验 LLM 的指令遵循稳定性。*

> 语言约定：`README.md` 为英文主入口；中文版本在 `README_ZH.md`。

---

## 0) 30 秒导航

1) **评测对象（题集 / schema / 协议）** → `01_experiment_design/README_ZH.md`

2) **生成侧提示词（A/B/… 变体清单）** → `02_prompt_variants/PROMPT_MANIFEST_ZH.md`

3) **评测规则（rubric + judge 契约）** → `03_evaluation_rules/EVAL_PROTOCOL_ZH.md` + `03_evaluation_rules/JUDGE_PROMPT_ZH.md`

4) **结果与快照位置（可审计）** → `04_results/README_ZH.md`

5) **解释层（结论边界 + 展望）** → `05_summary_and_outlook/README_ZH.md`

---

## 1) 这个仓库“是什么 / 不是什么”

### 是

- 以工件为中心的评测闭环：检验**提示词微小变化**下的**指令遵循稳定性**。
- 强证据链：任何总结/解读都应可回指到 `04_results/` 的工件与 `03_evaluation_rules/` 的协议。

### 不是

- 给模型“综合排名”的通用基准。
- 超出仓库内题集范围的泛化结论。

---

## 2) 目录结构（当前）

```
01_experiment_design/
  README.md / README_ZH.md
  eval_questions_EN.jsonl / eval_questions_ZH.jsonl
  output_schema.md / output_schema_ZH.md
  experiment_protocol.yaml / experiment_protocol_ZH.yaml
  terminology_alignment.md / terminology_alignment_ZH.md
  threats_and_limitations.md / threats_and_limitations_ZH.md

02_prompt_variants/
  README.md / README_ZH.md
  PROMPT_MANIFEST.md / PROMPT_MANIFEST_ZH.md
  00_baseline_prompt_A*.txt
  01_structured_prompt_B*.txt
  02_conflict_prompt*.txt
  (可选额外变体：03_long_prompt*, 04_weak_prompt*, ...)

03_evaluation_rules/
  README.md / README_ZH.md
  EVAL_PROTOCOL.md / EVAL_PROTOCOL_ZH.md
  JUDGE_PROMPT.md / JUDGE_PROMPT_ZH.md
  (辅助：validity_criteria*, scoring_dimensions*, compute_scores*.py, schema/)

04_results/
  README.md / README_ZH.md
  01_raw_model_outputs/                 # PDF：原始输出（按模型/题目/变体）
  02_cross_model_evaluation/
    valid_evaluations/
      main_method_cross_model/          # JSON：跨模型评测（主证据）
      supporting_method_self_eval/      # JSON：自评（支持性 sanity check）
      summary_tables/                   # CSV：用于分析的聚合表
    invalid_evaluations/                # 统计排除；保留用于审计/失败模式池
  03_results_analysis.md / 03_results_analysis_ZH.md

05_summary_and_outlook/
  README.md / README_ZH.md

06_methodological_addenda_and_controls/
  README.md / README_ZH.md
  A_B_comparative_rationale.md / A_B_comparative_rationale_ZH.md

07_deep_research/
  README.md（建议：补一个 README_ZH.md 以保持双语一致）
  *.pdf（文献/背景资料）
```

---

## 3) 最安全的结论边界（建议）

建议所有表述都写成：

- “在 **本题集** + **本提示词变体** + **本评测协议** 下，我们观察到 …”

避免：

- 扩展成跨任务/跨域的“总体模型能力”判断。

---

## 4) 可复现性备注

- `03_evaluation_rules/` 是评测口径的权威来源。
- 结果目录可保存当次使用的协议/清单快照（`used_evaluation_protocol*.md`、`used_prompt_manifest*.md`），用于审计与复跑。