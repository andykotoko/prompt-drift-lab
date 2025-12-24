# README.md (EN)

# Prompt Drift Lab
*A reproducible, artifact-first evaluation framework for instruction-following stability (prompt drift) under controlled prompt variants.*

> Language: **README.md is English-first**. A Chinese counterpart is provided as `README_ZH.md`.

---

## 0) 30-second start

1) **What was evaluated (questions / schema / protocol)?** → `01_experiment_design/README.md`

2) **What prompts were used (generator-side A/B/…)?** → `02_prompt_variants/PROMPT_MANIFEST.md`

3) **How scoring/judging works (rubric + judge contract)?** → `03_evaluation_rules/EVAL_PROTOCOL.md` + `03_evaluation_rules/JUDGE_PROMPT.md`

4) **Where are the results and the exact snapshots used?** → `04_results/README.md`

5) **So what (interpretation + claim boundary + outlook)?** → `05_summary_and_outlook/README.md`

---

## 1) What this repo is (and what it is not)

### This repo **is**

- An **artifact-centered** eval pipeline to test **instruction-following stability** when prompt wording/format constraints change.
- A **traceable evidence chain**: every claim in summaries should be traceable to `04_results/` artifacts and `03_evaluation_rules/` protocols.

### This repo **is not**

- A benchmark claiming global model rankings.
- A dataset release beyond the included small evaluation set.

---

## 2) Directory map (current)

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
  (optional additional variants: 03_long_prompt*, 04_weak_prompt*, ...)

03_evaluation_rules/
  README.md / README_ZH.md
  EVAL_PROTOCOL.md / EVAL_PROTOCOL_ZH.md
  JUDGE_PROMPT.md / JUDGE_PROMPT_ZH.md
  (supporting docs: validity_criteria*, scoring_dimensions*, compute_scores*.py, schema/)

04_results/
  README.md / README_ZH.md
  01_raw_model_outputs/                 # PDFs: raw outputs (per model / per question / per variant)
  02_cross_model_evaluation/
    valid_evaluations/
      main_method_cross_model/          # JSON: cross-model judging (main evidence)
      supporting_method_self_eval/      # JSON: self-eval (supporting sanity check)
      summary_tables/                   # CSV: aggregations used in analysis
    invalid_evaluations/                # excluded from stats; kept for audit/failure-mode pool
  03_results_analysis.md / 03_results_analysis_ZH.md

05_summary_and_outlook/
  README.md / README_ZH.md

06_methodological_addenda_and_controls/
  README.md / README_ZH.md
  A_B_comparative_rationale.md / A_B_comparative_rationale_ZH.md

07_deep_research/
  README.md (recommended: add README_ZH.md if you want bilingual)
  *.pdf (reading notes / background)
```

---

## 3) Safest claim boundary (recommended)

To keep the repo **workshop-safe and review-safe**, prefer claims of the form:

- “Under **this** question set + **these** prompt variants + **this** scoring protocol, we observe …”

Avoid:

- Broad generalization across tasks, domains, or “overall model quality”.

---

## 4) Reproducibility notes

- The canonical evaluation contract lives in `03_evaluation_rules/`.
- Results directories may snapshot the exact protocol/manifest used (e.g., `used_evaluation_protocol*.md`, `used_prompt_manifest*.md`) to keep runs auditable.