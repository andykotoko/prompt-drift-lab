# 03 Results Analysis

**You are here:** `04_results/03_results_analysis.md`  
**Upstream:** `01_experiment_design/` → `02_prompt_variants/` → `03_evaluation_rules/`  
**Downstream:** `05_summary_and_outlook/`  
**Sidecar:** `06_methodological_addenda_and_controls/` (controls & rationale), `07_deep_research/` (references)

## Purpose (What this document is responsible for)
This document explains **how to interpret the reported results** and provides a **failure-mode / attribution lens** for prompt drift.
It does **not** serve as a file index. For exact paths to `summary.csv`, raw outputs, and valid/invalid buckets, see:
- `04_results/README.md`

## Scope & claim boundary
- **Main quantitative reporting** is based on outputs generated with **Prompt B (protocol-ready)**.
- **Prompt A (exploratory)** is used only for **qualitative contrast** and mechanism hints.
- A/B numbers should **not** be merged unless **coverage and evaluation protocol are explicitly aligned**.

## What “Prompt Drift” means in this repo
Prompt drift refers to **systematic changes in model behavior** caused by small changes in prompt **format / wording / constraints** while keeping task semantics unchanged, typically observed as:
- Instruction-following drop
- Format/schema violations
- Semantic deviation

## Reading the results (recommended order)
1) **Check the aggregated table(s)** in `04_results/README.md` (where the numbers are)  
2) **Inspect representative samples** (raw outputs + judge rationales)  
3) **Use the failure-mode section below** to classify deviations and propose minimal ablations

## A vs B: why Prompt B is the main template
- **Prompt A (exploratory):** faster iteration, looser constraints, useful for collecting failure cases early.
- **Prompt B (protocol-ready):** stronger structure anchoring and enforceable constraints, improving cross-sample comparability.

> Design choice rationale and any control comparisons should be documented in `06_methodological_addenda_and_controls/`.

## Invalid evaluations & failure modes (used for diagnosis, not for scoring)
Some runs are excluded from quantitative aggregation and stored as **invalid evaluations**. These samples are used to:
- characterize failure modes,
- explain non-comparability,
- drive protocol/prompt improvements **without changing existing conclusions**.

### Failure taxonomy (covers the core drift types)
A. **Schema / format errors**: missing fields, wrong order/type, extra text, broken JSON/structure  
B. **Instruction deviation**: missed requirements, ignored constraints, unsolicited steps  
C. **Semantic drift**: wrong answer, missing key info, contradictions/hallucinations  
D. **Robustness / variance**: high run-to-run instability under the same prompt  
E. **Rubric gaming**: keyword compliance without solving the task

### Suggested flags (for invalid buckets)
Use flags to keep diagnosis consistent across versions:
- `PROTOCOL_VIOLATION`, `UNPARSABLE_OUTPUT`, `INCOMPLETE_COVERAGE`
- `JUDGE_REFUSAL_OR_EVASION`, `INTERNAL_INCONSISTENCY`, `CONTEXT_MISALIGNMENT`
- `SELF_JUDGING_BIAS` (supporting only; not headline evidence)

## Attribution playbook (minimal ablations)
When you observe drift, document it as:
1) **Locate**: which failure type(s) A–E?  
2) **Hypothesize**: which mechanism (ambiguity, priority conflict, weak constraints, model preference)?  
3) **Ablate**: change **one** factor only (delimiter/heading/neg constraint/length)  
4) **Count**: report frequency (and variance if repeated runs are available)

## Reproducibility checklist (what must be traceable)
For every claim you write in `05_summary_and_outlook/`, ensure you can point to:
- a row in the summary table (or an aggregation rule),
- the corresponding raw outputs,
- the judged scores + rationale snippets,
- the exact prompt version / judge prompt id / run config.
