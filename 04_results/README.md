# 04 Results

This folder contains **all experiment artifacts** produced by running the evaluation pipeline:
- raw model outputs (PDFs)
- judged outputs (JSON)
- aggregated tables (CSV)
- short narrative analysis

If you want the **interpretation** (A/B rationale, drift attribution, failure modes), read:
- `04_results/03_results_analysis.md`
- and, for methodological rationale / controls: `06_methodological_addenda_and_controls/`

---

## 0) 30-second navigation (start here)

1) **Top-line summary (CSV)**
- `04_results/02_cross_model_evaluation/valid_evaluations/summary_tables/summary.csv`

2) **Breakdowns (CSV)**
- Main method (cross-model judging):
  - `.../summary_tables/main_method_by_generator.csv`
  - `.../summary_tables/main_method_by_version.csv`
  - `.../summary_tables/main_method_by_question.csv`
  - `.../summary_tables/main_method_by_question_version.csv`
  - `.../summary_tables/main_method_inter_judge_agreement.csv`
- Supporting method (self-eval sanity check):
  - `.../summary_tables/supporting_method_by_generator.csv`
  - `.../summary_tables/supporting_method_by_version.csv`
  - `.../summary_tables/supporting_method_by_question.csv`
  - `.../summary_tables/supporting_method_by_question_version.csv`

3) **Judge outputs (JSON)**
- Valid:
  - Main method: `04_results/02_cross_model_evaluation/valid_evaluations/main_method_cross_model/`
  - Supporting method: `04_results/02_cross_model_evaluation/valid_evaluations/supporting_method_self_eval/`
- Invalid (excluded from stats; used for audit / failure-mode pool):
  - `04_results/02_cross_model_evaluation/invalid_evaluations/`

4) **Raw model outputs (PDF)**
- `04_results/01_raw_model_outputs/<generator_model>/`

---

## 1) Directory map

