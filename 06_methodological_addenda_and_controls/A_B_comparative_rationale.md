# A/B Comparative Rationale: Introducing A vs. B and Using B as the Anchor

This document addresses two core questions:

- **Why an explicit A/B comparison is necessary**;
- **Why subsequent perturbations (long, weak, conflict) are all defined with Prompt B as the anchor**.

All arguments below are strictly based on **experiments that have already been completed and materialized** in this project. No unimplemented methods or assumptions are introduced.

---

## 1. Minimal Difference Definition Between Prompt A and Prompt B

The difference between Prompt A and Prompt B is intentionally restricted to **a single central axis: the degree of structure and constraint strength**.

- **Prompt A**: primarily natural-language–driven, with weak structural boundaries and relatively loose constraints on output format.
- **Prompt B**: introduces explicit fields, ordering requirements, and prohibited items, imposing hard, executable constraints on output structure.

The A/B comparison follows these control principles:

- No new task content is introduced;
- The question set, model parameters, and judge protocol remain unchanged;
- Differences are confined to the **executability** and **verifiability** of the output structure.

The corresponding audit entry is available at: `02_prompt_variants/PROMPT_MANIFEST.md`.

---

## 2. Why an Explicit A/B Comparison Is Necessary

Without an explicit A/B comparison, the reference frame for prompt drift becomes ambiguous, for example:

- When format breakdown occurs, it is unclear whether the cause lies in model capability or insufficient structural specification in the prompt;
- The degree of degradation introduced by subsequent perturbations lacks a well-defined baseline;
- When the invalid rate increases, it is difficult to determine whether this is caused by the perturbation itself or by an already non-evaluable baseline.

The A/B comparison establishes a minimal coordinate system:

- **Prompt A** represents a weakly constrained, more natural interaction setting;
- **Prompt B** represents a structured, evaluable anchor suitable for alignment and comparison.

---

## 3. Rationale for Selecting Prompt B as the Anchor

### 3.1 Evaluability

When output structure is unstable, samples are more likely to fall into the invalid set, causing statistical results to be dominated by formatting noise. The constraint design of Prompt B aims to increase the proportion of **evaluable samples**, allowing evaluation to focus on instruction following and semantic deviation rather than structural failure.

### 3.2 Interpretability of the Perturbation Space

With Prompt B as the anchor, subsequent perturbations can be interpreted as **directional modifications within the same constraint system**:

- **weak**: systematically reduces structural and hard constraints;
- **long**: increases contextual load and introduces interference while keeping constraints unchanged;
- **conflict**: introduces instruction tension to probe implicit priority resolution.

### 3.3 Cross-Model Comparability

When output structure is more stable, different judge models are more likely to align on the same evaluation targets, reducing evaluation variance introduced by formatting noise and improving cross-model comparability.

---

## 4. Organization of the Evidence Chain

In this project, evidence related to the A/B comparison is organized in the following order:

1. Use `04_results/summary.csv` to locate differences between A and B in metrics such as hard compliance and invalid rate;
2. Sample corresponding PDF outputs to confirm that differences arise primarily from structural adherence rather than incidental content variation;
3. Inspect the associated judge JSON outputs to verify that invalid labels and scoring rationales conform to the evaluation protocol;
4. On this basis, analyze the weak, long, and conflict perturbations built on Prompt B to verify that relative degradation directions align with observed failure-mode mappings.

---

## 5. Limitations and Risk Control

- Prompt A and B may still differ in secondary aspects such as length or tone; these effects should be constrained through controlled experiments or ablation when possible;
- Model-specific differences in executability of structured instructions mean that the optimal anchor choice may vary across models;
- Automated judges may exhibit structural preferences; such bias can be mitigated through cross-model judging, self-judging, spot checks, or judge swapping.

