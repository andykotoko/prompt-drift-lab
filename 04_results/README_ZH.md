# 04_results/README_ZH.md (ZH)

# 04 实验结果

本目录存放一次评测流水线产出的**全部结果工件**：
- 原始模型输出（PDF）
- 评测输出（JSON）
- 汇总表（CSV）
- 简要结果分析（markdown）

如果你要看**解释与归因**（A/B 选择依据、漂移机制、失败模式），请读：
- `04_results/03_results_analysis_ZH.md`
- 方法学补充与对照：`06_methodological_addenda_and_controls/`

---

## 0) 30 秒导航（从这里开始）

1) **总览汇总（CSV）**
- `04_results/02_cross_model_evaluation/valid_evaluations/summary_tables/summary.csv`

2) **分组统计（CSV）**
- 主方法（跨模型互评）：
  - `.../summary_tables/main_method_by_generator.csv`
  - `.../summary_tables/main_method_by_version.csv`
  - `.../summary_tables/main_method_by_question.csv`
  - `.../summary_tables/main_method_by_question_version.csv`
  - `.../summary_tables/main_method_inter_judge_agreement.csv`
- 辅助方法（自评 sanity check）：
  - `.../summary_tables/supporting_method_by_generator.csv`
  - `.../summary_tables/supporting_method_by_version.csv`
  - `.../summary_tables/supporting_method_by_question.csv`
  - `.../summary_tables/supporting_method_by_question_version.csv`

3) **评测输出（JSON）**
- 有效（进入统计）：
  - 主方法：`04_results/02_cross_model_evaluation/valid_evaluations/main_method_cross_model/`
  - 辅助方法：`04_results/02_cross_model_evaluation/valid_evaluations/supporting_method_self_eval/`
- 无效（不进统计；用于审计/失败模式样本池）：
  - `04_results/02_cross_model_evaluation/invalid_evaluations/`

4) **原始模型输出（PDF）**
- `04_results/01_raw_model_outputs/<generator_model>/`

---

## 1) 目录地图

```
04_results/
  01_raw_model_outputs/                  # PDF：按模型/题目/变体保存
  02_cross_model_evaluation/
    valid_evaluations/
      main_method_cross_model/           # JSON：跨模型互评（主证据）
      supporting_method_self_eval/       # JSON：自评（仅 sanity check）
      summary_tables/                    # CSV：用于分析的汇总表
    invalid_evaluations/
      main_method_cross_model/           # JSON：无效（主方法）
      supporting_method_self_eval/       # JSON：无效（辅助方法）
      used_evaluation_protocol*.md       # 当次运行使用的协议快照（若有）
      used_prompt_manifest*.md           # 当次运行使用的 prompt 清单快照（若有）
      README.md                          # 无效样本如何阅读
  03_results_analysis.md                 # 结果分析（EN）
  03_results_analysis_ZH.md              # 结果分析（ZH）
```

---

## 2) 什么是“主证据”/“辅助证据”/“无效池”（口径边界）

- **主证据（报告/统计使用）**：
  `valid_evaluations/main_method_cross_model/` + `summary_tables/*main_method*`
- **辅助证据（仅 sanity check）**：
  `valid_evaluations/supporting_method_self_eval/` + `summary_tables/*supporting_method*`
- **无效池（绝不参与统计）**：
  `invalid_evaluations/` 下所有内容

---

## 3) 复现指引（指向权威规则，不在本文件复述）

- 评测规则权威入口：`03_evaluation_rules/`
- judge 契约：`03_evaluation_rules/JUDGE_PROMPT_ZH.md`
- 若某次运行在结果目录中落了“当次协议/清单快照”，会以：
  `used_evaluation_protocol*.md` 与 `used_prompt_manifest*.md` 的形式出现（以实际文件为准）。