# Judge Prompts Inventory

This document **systematically records** all judge prompts actually used during the **evaluation stage** of the Prompt Drift Lab project, in order to ensure:

- **Traceability**: every evaluation result can be traced back to the exact judge prompt that produced it;
- **Reproducibility**: third parties can rerun the evaluation under the same protocol and obtain comparable results;
- **Auditability**: when an evaluation is deemed invalid, it can be clearly attributed to either *protocol execution failure* or *judge prompt design*.

> **Scope note**: This inventory **only covers evaluation-stage prompts (judge prompts)**.
> Prompts used during the generation stage are **explicitly excluded** and must be documented separately in their corresponding directories.

---

## 0. Naming and Referencing Conventions

- Each judge prompt is assigned a **stable and unique identifier** in the form `JP-XXX` (Judge Prompt).
- Every prompt entry must explicitly declare the following metadata fields:
  - `method`: evaluation method, one of `cross_judge` or `self_judge`;
  - `protocol_file`: the evaluation protocol file used (located in this directory);
  - `expected_output`: the strictly required output format (uniformly a **single JSON object** in this project);
  - `scope`: the applicability of the prompt (e.g., all questions / specific `question_id` / specific output types).

It is **strongly recommended** to include the following fields in the `meta` object of each evaluation result JSON:

- `judge_prompt_id`: `JP-001`
- (optional) `protocol_file`: `EVAL_PROTOCOL_ZH.md`

---

## 1. Primary Method: Cross-Model Judging

### JP-001 (Cross Judge / General)

- **method**: `cross_judge`
- **protocol_file**: `./EVAL_PROTOCOL_ZH.md`
- **expected_output**: output **one and only one JSON object** (no Markdown, no explanatory text, no prefixes or suffixes)
- **scope**: applicable to all `question_id`, all `prompt_variant`, and all `target_model`

**Prompt (full text)**:

```
You are a strict evaluation model (judge).
You will receive:
(1) the output text produced by the target model;
(2) the corresponding metadata (question_id, prompt_variant, target_model, output_id).

Your tasks:
1) Evaluate the output strictly according to `EVAL_PROTOCOL_ZH.md`:
   - Dimensions: FORMAT_COMPLIANCE / INSTRUCTION_COMPLIANCE / SEMANTIC_FIDELITY / COMPLETENESS
   - Each dimension is scored as an integer: 0 / 1 / 2
   - overall_score = the sum of the four dimensions
2) Produce a verdict: PASS / PARTIAL / FAIL (according to protocol-defined thresholds)
3) For each dimension, provide at least one piece of evidence (a short quote + one-sentence justification)
4) If any of the following occurs:
   - protocol violation
   - unparsable output
   - incomplete coverage
   - judge refusal or evasion
   - internal inconsistency
   then you must populate the corresponding labels in `flags`:
   PROTOCOL_VIOLATION / UNPARSABLE_OUTPUT / INCOMPLETE_COVERAGE /
   JUDGE_REFUSAL_OR_EVASION / INTERNAL_INCONSISTENCY

Output requirements (critical):
- You must output **exactly one JSON object** and nothing else.
- No Markdown, no explanations, no surrounding text.
- The JSON must contain all required fields: meta / scores / verdict / flags / evidence / notes.
- All four score dimensions must be present and must be integers in {0,1,2}.

Now begin evaluation. The input is as follows:
[meta]
{META_JSON}

[target_output]
{TARGET_OUTPUT}
```

---

## 2. Auxiliary Method: Self Judging

### JP-002 (Self Judge / General)

- **method**: `self_judge`
- **protocol_file**: `./EVAL_PROTOCOL_ZH.md`
- **expected_output**: output **one and only one JSON object** (no Markdown, no explanatory text, no prefixes or suffixes)
- **scope**: applicable to all `question_id`, all `prompt_variant`, and all `target_model`

**Prompt (full text)**:

```
You are a strict evaluation model (judge).
You will evaluate your own previously generated output.

You will receive:
(1) your output text;
(2) the corresponding metadata (question_id, prompt_variant, target_model, output_id).

Your tasks:
1) Evaluate the output strictly according to `EVAL_PROTOCOL_ZH.md`:
   - Dimensions: FORMAT_COMPLIANCE / INSTRUCTION_COMPLIANCE / SEMANTIC_FIDELITY / COMPLETENESS
   - Each dimension is scored as an integer: 0 / 1 / 2
   - overall_score = the sum of the four dimensions
2) Produce a verdict: PASS / PARTIAL / FAIL (according to protocol-defined thresholds)
3) For each dimension, provide at least one piece of evidence (a short quote + one-sentence justification)
4) If any of the following occurs:
   - protocol violation
   - unparsable output
   - incomplete coverage
   - judge refusal or evasion
   - internal inconsistency
   then you must populate the corresponding labels in `flags`:
   PROTOCOL_VIOLATION / UNPARSABLE_OUTPUT / INCOMPLETE_COVERAGE /
   JUDGE_REFUSAL_OR_EVASION / INTERNAL_INCONSISTENCY

Output requirements (critical):
- You must output **exactly one JSON object** and nothing else.
- No Markdown, no explanations, no surrounding text.
- The JSON must contain all required fields: meta / scores / verdict / flags / evidence / notes.
- All four score dimensions must be present and must be integers in {0,1,2}.

Now begin evaluation. The input is as follows:
[meta]
{META_JSON}

[self_output]
{TARGET_OUTPUT}
```

---

## 3. Variables and Metadata Template (for Direct Use)

```json
{
  "judge_model": "...",
  "target_model": "...",
  "question_id": "Q1",
  "prompt_variant": "A",
  "output_id": "...",
  "method": "cross_judge",
  "judge_prompt_id": "JP-001",
  "protocol_file": "EVAL_PROTOCOL_ZH.md",
  "timestamp": "YYYY-MM-DD"
}
```

- `judge_prompt_id` must strictly correspond to a `JP-XXX` entry defined in this inventory.
- Optional fields may include (but are not limited to): `run_id`, `seed`, `temperature`, `top_p`, etc., for full experimental reproducibility.

