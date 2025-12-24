# Methodological Addenda and Controls (Prompt Drift Lab)

This directory provides **methodological justification and design constraints** for the evaluation setup, ensuring fair comparisons, attributable conclusions, and auditable experimental artifacts.

Importantly, this directory **does not introduce any new evaluation results**. Its purpose is to clarify:

- how the comparison framework in this repository is constructed;
- whether the comparisons satisfy minimal-difference and fairness requirements;
- how conclusions can be audited, verified, and extended without violating existing evidence boundaries.

---

## 1. Responsibilities of This Directory

In engineering-style evaluations of prompt drift, methodological concerns typically fall into three categories:

- **Fairness of comparisons**: whether A/B constitutes a minimal difference and whether unintended variables are introduced;
- **Attribution of conclusions**: whether observed differences arise from prompt perturbations themselves rather than sampling variance, judge bias, or post-processing strategies;
- **Validity of the main anchor**: why Prompt B is selected as the baseline for subsequent perturbations instead of Prompt A.

This directory provides unified, auditable answers to these questions, along with executable checks for verification.

---

## 2. Positioning of the A/B Comparison in This Repository

In this repository, prompt variants are assigned **distinct experimental roles**, rather than being treated as simply “better” or “worse” prompts:

- **Prompt A**: more natural-language–oriented, with weaker structural constraints, serving as a lower-bound reference approximating real-world weakly constrained interactions;
- **Prompt B**: more strongly structured, with explicit fields, ordering requirements, and prohibited items, serving as the anchor for the perturbation space.

The purpose of the A/B comparison is not to claim that structured prompts are inherently superior, but to **establish a reproducible and quantifiable coordinate system** in which prompt drift can be consistently observed and interpreted.

Relevant entry points include:

- Prompt versions and differences: `02_prompt_variants/PROMPT_MANIFEST.md`
- Evaluation protocol and judge prompts: `03_evaluation/EVAL_PROTOCOL.md`, `03_evaluation/JUDGE_PROMPT.md`
- Results and evidence: `04_results/`

---

## 3. Criteria for Selecting Prompt B as the Main Perturbation Anchor

This repository adopts **auditable engineering criteria** for anchor selection, rather than subjective preference.

### 3.1 Evaluability as a Primary Criterion

When a prompt variant frequently triggers schema violations and pushes a large fraction of samples into the invalid set, statistical analysis becomes dominated by formatting noise, making it difficult to reason about semantic drift or constraint satisfaction. One design objective of Prompt B is therefore to **increase the proportion of evaluable samples**.

### 3.2 Interpretability of the Perturbation Space

With Prompt B as the anchor, subsequent perturbations can be interpreted as **single-axis modifications within the same constraint system**:

- **weak**: systematically reduces structural and hard constraints;
- **long**: increases contextual load and introduces potential interference;
- **conflict**: introduces instruction tension to probe priority-handling behavior.

This design allows observed drift to be mapped to specific failure types and potentially actionable issues, rather than remaining a generic notion of instability.

### 3.3 Cross-Model Comparability

When output structure is more stable, judge models are more likely to align on the same evaluation targets, reducing scoring variance induced by formatting noise and improving cross-model comparability.

---

## 4. Checklist for Fairness and Auditability

To ensure fair comparisons and auditable conclusions, this repository adheres to the following checks:

- Core A/B differences are restricted to structural degree and constraint strength, while task content remains unchanged;
- Question sets, target models, inference parameters, and judge protocols are held constant across comparison groups;
- Raw outputs and judge JSON files are preserved on a per-sample basis, enabling traceability from summary tables back to evidence;
- Invalid labels follow protocol-defined criteria and retain supporting rationale excerpts;
- Statistical scripts are aligned with the evaluation protocol, with consistent field definitions and aggregation rules;
- Where feasible, minimal repeated runs are conducted to observe variance; otherwise, limitations are explicitly disclosed.

---

## 5. Mapping Perturbation Types to Failure Modes

For analysis and review, failure modes are treated as the primary organizing principle, rather than post hoc narrative fitting:

- **weak**: more likely to exhibit format drift and hard-constraint omission;
- **long**: more likely to induce attention dilution, implicit-constraint failure, and key-information omission;
- **conflict**: more likely to trigger priority-resolution failures and model-initiated compromise strategies.

All conclusions should be traced back to the evidence chain, including corresponding PDF outputs and scoring rationales in judge JSON files.

---

## 6. Limitations and Risk Control

- A/B comparisons may still involve secondary differences such as prompt length or tone; interpretive claims should therefore be carefully scoped, and mitigated through controlled comparisons or ablation where possible;
- The optimal anchor choice may vary across models; cross-model conclusions must clearly state their applicability boundaries;
- Automated judges may exhibit structural preferences; such bias can be reduced through cross-model judging, self-judging, spot checks, or judge rotation.

