# Evaluation Protocol Used (Judge Protocol)

This document defines the **unified evaluation protocol** used in this project for both **cross-model judging** and **self-judging**. Its goal is to ensure that evaluation results are **machine-parsable, aligned, and auditable**, and to clearly distinguish between **valid evaluations** and **invalid evaluations**.

---

## 0. Evaluation Targets and Units

- **Evaluation target**: Model outputs generated for the same question under different prompt variants.
- **Evaluation unit (sample)**:
  - `question_id` (e.g., Q1–Q4)
  - `prompt_variant` (e.g., A/B, or long/short, explicit/implicit as defined in the repository)
  - `target_model` (the model that produced the output being evaluated)
  - `output_id` (filename or unique identifier used to trace back to the raw output)

> Note: This protocol specifies *how evaluation is performed and how results are recorded*. It does **not** prescribe how questions or prompts should be designed.

---

## 1. Evaluation Methods

### 1.1 Primary Method: Cross-Model Judging

- A judge model evaluates outputs produced by *other* models.
- Purpose: to approximate a third-party or user perspective and reduce self-evaluation bias.

### 1.2 Supporting Method: Self-Judging

- A model evaluates its own outputs using the same protocol.
- Purpose: diagnostic comparison, protocol sanity checks, and failure-mode exploration.
- Limitation: self-judging results are **not used as primary statistics**, unless explicitly stated in the results analysis.

---

## 2. Evaluation Dimensions (Rubric)

This project focuses on failures induced by small prompt perturbations, including instruction-following degradation, format breakage, and semantic drift. To enable cross-model alignment, the following shared dimensions are used:

### D1. Structure / Format Compliance (`FORMAT_COMPLIANCE`)

- Whether the output follows the required structure (e.g., Markdown/JSON template, sectioning, headings, required fields).

### D2. Instruction Compliance (`INSTRUCTION_COMPLIANCE`)

- Whether key constraints are followed (e.g., mandatory three-part structure, required fields, forbidden content, evidence requirements).

### D3. Semantic Fidelity / Drift (`SEMANTIC_FIDELITY`)

- Alignment with the task intent and prompt goal; whether task rewriting, off-topic responses, or generic advice occurs.

### D4. Completeness (`COMPLETENESS`)

- Whether all required components are present; missing sections, skipped items, or partial responses are penalized.

> Optional: If a finer-grained rubric (e.g., 8 dimensions) exists, it may be mapped onto these four dimensions during aggregation (e.g., format break → D1, instruction following → D2, semantic drift → D3, missing fields → D4).

---

## 3. Scoring Scale

Each dimension is scored on a **discrete 0–2 scale** to reduce subjective noise:

- **2 = Satisfied**: Clearly meets the requirement; minor flaws do not affect parsing or verification.
- **1 = Partially satisfied**: Noticeable deviation, but intent or partial structure remains recognizable.
- **0 = Not satisfied**: Dimension fails (e.g., structure collapse, key constraints ignored, severe semantic drift, major omissions).

Additionally:

- `overall_score`: Sum of the four dimension scores (range 0–8).
- `verdict`: `PASS` / `PARTIAL` / `FAIL` (based on thresholds below).

### Recommended Thresholds (fixed within one experiment cycle)

- `PASS`: overall ≥ 7 **and** D1, D2 are not 0
- `PARTIAL`: overall 4–6, or any dimension equals 0
- `FAIL`: overall ≤ 3

> Thresholds are not absolute truths, but must remain fixed within a single experiment cycle.

---

## 4. Evidence Requirements

Each evaluation must include **verifiable evidence snippets** supporting the assigned scores:

- At least **one evidence item per dimension** (snippets may overlap across dimensions).
- Evidence format:
  - `quote`: a short excerpt from the evaluated output (recommended ≤ 30–60 words/characters)
  - `reason`: a brief explanation of why the excerpt supports the score

> For PDF or long-text outputs, evidence only needs to be *locatable and checkable*; line numbers are not required.

---

## 5. Output Format (Strict JSON)

The judge must output **exactly one JSON object**, with no Markdown, explanations, or extra text.

```json
{
  "meta": {
    "judge_model": "...",
    "target_model": "...",
    "question_id": "Q3",
    "prompt_variant": "A",
    "output_id": "...",
    "method": "cross_judge",
    "timestamp": "YYYY-MM-DD"
  },
  "scores": {
    "FORMAT_COMPLIANCE": 0,
    "INSTRUCTION_COMPLIANCE": 0,
    "SEMANTIC_FIDELITY": 0,
    "COMPLETENESS": 0,
    "overall_score": 0
  },
  "verdict": "FAIL",
  "flags": [
    "PROTOCOL_VIOLATION",
    "UNPARSABLE_OUTPUT"
  ],
  "evidence": [
    {"dimension": "FORMAT_COMPLIANCE", "quote": "...", "reason": "..."}
  ],
  "notes": "(optional)"
}
```

### Field Constraints

- All four dimension scores must be present and be integers in {0,1,2}.
- `overall_score` must equal the sum of the four dimensions.
- `flags` must be present (may be an empty array `[]`).
- `method` must be either `cross_judge` or `self_judge`.

---

## 6. Invalid Evaluation Criteria

An evaluation is classified as **invalid** and must be placed under `invalid_evaluations/` if **any** of the following occurs:

- **PROTOCOL\_VIOLATION**: Output does not follow this protocol, or rubric definitions/scales are altered.
- **UNPARSABLE\_OUTPUT**: JSON syntax errors, missing fields, type mismatches, or parse failures.
- **INCOMPLETE\_COVERAGE**: Missing key identifiers (`question_id`, `prompt_variant`, `target_model`, `output_id`), or misaligned evaluation sets.
- **JUDGE\_REFUSAL\_OR\_EVASION**: Refusal to score, missing scores, or irrelevant output.
- **INTERNAL\_INCONSISTENCY**: e.g., `overall_score` does not equal the sum of dimension scores, or irreconcilable contradictions.

> Note: *Invalid* does not mean *low quality*. Low scores correspond to `FAIL`, not to invalid evaluations.

---

## 7. Recommended Execution Workflow

1. **Define the evaluation set**: enumerate all `output_id`s aligned by question × variant × target\_model.
2. **Run the judge**: evaluate each sample once, forcing strict JSON output.
3. **Structural validation**:
   - JSON syntax correctness
   - Field completeness
   - Score range validity (0/1/2)
   - `overall_score` consistency
4. **Archiving**:
   - Pass validation → `valid_evaluations/...`
   - Fail validation or invalid criteria → `invalid_evaluations/...` (with flags)

---

## 8. Reproducibility Metadata (Recommended)

For each evaluation run, record the following in run logs:

- Judge model and version (if available)
- Decoding parameters (temperature, top\_p, max\_tokens)
- Tool or web access (if enabled)
- Evaluation time range and sample count

---

## 9. Relationship to Other Documents

- `used_prompt_manifest.md`: lists the judge prompts actually used (cross-model and self-judging).
- `invalid_evaluations/README.md`: defines invalid boundaries and directory semantics.
- `04_results/03_results_analysis.md`: analyzes failure modes and mechanisms *after* facts are frozen.

